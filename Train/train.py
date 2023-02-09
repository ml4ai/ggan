"""
Training script to experiment with single obj files: to make sure the model can fit on a single 3d object
and to make sure each components are working as expected
"""

import argparse
import json
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from Utils.utils import weights_init
from torch import nn
from torch import optim
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, SoftPhongShader, TexturesUV
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer import HardFlatShader
from pytorch3d.datasets import ShapeNetCore
import warnings
from Utils.plot import plot_original_and_generated
from Models.encoder import Encoder
from PIL import Image
from torchvision import transforms
from random import random
from StyleGAN2.stylegan2 import StyleGAN2, mixed_list, noise_list
from StyleGAN2.stylegan2 import image_noise, latent_to_w, styles_def_to_tensor
from Utils.utils import d_logistic_loss, g_nonsaturating_loss

warnings.filterwarnings("ignore")

# device to train: gpu/cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parse the arguments for config of Generator and config of Discriminator
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration file for paths and hyperparameters")
parser.add_argument("--configG", help="configuration file for Generator")
parser.add_argument("--configD", help="configuration file for Discriminator")
args = parser.parse_args()

config_file = args.config
config = json.load(open(config_file, "r"))

# batch size used for training
batch_size = config["batch_size"]
# different settings for shapenet rendering
is_shapenet = config["is_shapenet"]
# in case of shapenet - specify category
category = config["category"]

# option to either use uv_layout encoder or not
use_uv_layout_encoder = config["use_uv_layout_encoder"]

# save path for trained generator
save_path_generator = config["save_path_generator"]
save_path_encoder = config["save_path_encoder"]

# if debug: will show the image rendered with original texture and rendered with synthetic texture during training
debug = config["debug"]

# Learning rate for optimizers
lr = config["lr"]

# Beta1 hyperparam for Adam optimizers
beta1 = config["beta1"]

# set gan_only just to train from gan losses
gan_only = config["gan_only"]

# train with multiple losses: gan loss, l1 loss between image rendered with synthetic texture and
# rendered with original texture
# weight for l1 loss: weight for l1 loss
l1_weight = config["l1_weight"]
# weight for gan loss: weight for gan loss
gan_weight = config["gan_weight"]

# spatial size of image generated from generator/ input to Discriminator
image_size = config["image_size"]

stylegan2 = config["stylegan2"]

if stylegan2:
    lr = config["stylegan2_lr"]
    lr_mlp = config["lr_mlp"]
    ttur_mult = config["ttur_mult"]
    network_capacity = config["network_capacity"]
    fmap_max = config["fmap_max"]
    latent_dim = config["latent_dim"]
    style_depth = config["style_depth"]
    mixed_prob = config["mixed_prob"]

    GAN = StyleGAN2(image_size=image_size, latent_dim=latent_dim, fmap_max=fmap_max, style_depth=style_depth,
                    network_capacity=network_capacity,
                    lr=lr, lr_mlp=lr_mlp, ttur_mult=ttur_mult)

    num_layers = GAN.G.num_layers

    print("Initializing StyleVectorizer...")
    netS = GAN.S.to(device)
    print(netS)

    print("Initializing Generator...")
    netG = GAN.G.to(device)
    print(netG)

    print("Initializing Discriminator...")
    netD = GAN.D.to(device)
    print(netD)

    # optimizer for generator
    optimizerG = GAN.G_opt
    # optimizer for discriminator
    optimizerD = GAN.D_opt
else:
    if image_size == 64:
        from Generator.generator64 import Generator
        from Discriminator.discriminator64 import Discriminator
    elif image_size == 128:
        from Generator.generator128 import Generator
        from Discriminator.discriminator128 import Discriminator
    elif image_size == 512:
        from Generator.generator512 import Generator
        from Discriminator.discriminator512 import Discriminator
    else:
        from Generator.generator1024 import Generator
        from Discriminator.discriminator1024 import Discriminator

    # netG and netD
    print("Initializing Generator...")
    netG = Generator(args.configG).to(device)
    print(netG)

    print("Initializing Discriminator...")
    netD = Discriminator(args.configD).to(device)
    print(netD)

    print("Initializing weights for Generator and Discriminator...")
    netG.apply(weights_init)
    netD.apply(weights_init)

    # optimizer for Discriminator
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    # optimizer for Generator
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

configG = json.load(open(args.configG, "r"))
nz = configG["nz"]
out_dim = nz // 2
nc = configG["nc"]
# encoder
encoder = Encoder(in_channels=nc, out_dim=out_dim).to(device) if use_uv_layout_encoder else None

# num epochs to train
num_epochs = config["num_epochs"]

# if shapenet: provide the shapenet paths
shapenet_path = config["shapenet_path"] if is_shapenet else None
shapenet_dataset = ShapeNetCore(shapenet_path, synsets=[category], version=2, load_textures=True,
                                texture_resolution=4) if is_shapenet else None

# if not shapenet: provide the obj path
if not is_shapenet:
    obj_path = config["obj_path"]

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

# optimizer for Encoder
optimizerE = optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, 0.999)) if use_uv_layout_encoder else None

# loss for generator
loss_G = []
# loss for discriminator
loss_D = []
# loss for Image
loss_I = []

# use approximate mesh if true
mesh_approximation = config["mesh_approximation"]

print("Training...")
for epoch in range(num_epochs):
    # for each epoch iterate over the obj files and train
    print(f"Epoch: {epoch + 1} / {num_epochs}")
    # load obj file
    shapenet_model = shapenet_dataset[0] if is_shapenet else None
    # if mesh approximation is true: use approximation mesh and use the uv layout of icosphere for all models
    if is_shapenet:
        # get some information
        synset_id = shapenet_model["synset_id"]
        model_id = shapenet_model["model_id"]
        if mesh_approximation:
            obj_path = config["approx_mesh_path"]
        else:
            obj_path = shapenet_path + "/" + synset_id + "/" + model_id + "/models/" + "bpy_model.obj"
        uv_layout_path = shapenet_path + "/" + synset_id + "/" + model_id + "/models/uv_layout.png" if use_uv_layout_encoder else None

        verts, faces, aux = load_obj(obj_path, device=device)
        mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])
        # load the mesh with uv information and apply generated texture
        faces_uvs = [faces.textures_idx]
        # verts_uvs = mesh.textures._verts_uvs_list
        verts_uvs = [aux.verts_uvs]
    else:
        mesh = load_objs_as_meshes([obj_path], device=device)
        faces_uvs = mesh.textures.faces_uvs_list()
        verts_uvs = mesh.textures.verts_uvs_list()

    # normalize and center the target mesh so that we can have same light and similar R, T for camera
    # that works on every objects
    # shapenet data is normilized - no need to normalize them
    if not is_shapenet:
        verts = mesh.verts_packed()
        N = verts.shape[0]
        center = verts.mean(0)
        scale = max((verts - center).abs().max(0)[0])
        mesh.offset_verts_(-center.expand(N, 3))
        mesh.scale_verts_((1.0 / float(scale)))
    # sample new camera location
    # distance = round(random.uniform(1.0, 2.0), 2)
    distance = 1.0
    elevation = torch.FloatTensor(batch_size).uniform_(-10, 180)
    azimuth = torch.FloatTensor(batch_size).uniform_(-180, 180)

    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation, azimuth=azimuth,
                                                            degrees=True, device=str(device))

    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    if is_shapenet:
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                cull_backfaces=True)
    else:
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                cull_backfaces=False)

    # Place a point light in front of the object: for now let's put at the same location of camera: so the rendered image will get some
    # light, we can also place them at s*camera_location where s is some scalar
    lights = PointLights(location=camera_location, ambient_color=((0.5, 0.5, 0.5),), diffuse_color=((0.4, 0.4, 0.4),),
                         specular_color=((0.1, 0.1, 0.1),), device=str(device))

    # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                            shader=HardFlatShader(cameras=cameras, lights=lights, device=device) if is_shapenet else
                            SoftPhongShader(device=device, cameras=cameras, lights=lights))

    # render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
    if is_shapenet:
        rendered_images = shapenet_dataset.render(model_ids=[model_id], device=device, cameras=cameras,
                                                  raster_settings=raster_settings,
                                                  lights=lights, shader_type=HardFlatShader)
    else:
        rendered_images = renderer(mesh, cameras=cameras, lights=lights)

    # zero grad the netD
    netD.zero_grad()
    # extract RGB images from these real rendered images
    real_batch = rendered_images[..., :3].to(device).permute(0, 3, 1, 2)
    label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
    # forward pass the real batch through D
    if stylegan2:
        real_pred = netD(real_batch)
    else:
        output = netD(real_batch).view(-1)
        errD_real = criterion(output, label)
        # calculate gradients for real batch
        errD_real.backward()

    if stylegan2:
        get_latents_fn = mixed_list if random() < mixed_prob else noise_list
        style = get_latents_fn(n=batch_size, layers=num_layers, latent_dim=latent_dim, device=device)
        noise = image_noise(n=batch_size, im_size=image_size, device=device)

        w_space = latent_to_w(netS, style)
        w_styles = styles_def_to_tensor(w_space)

        generated_images = netG(w_styles, noise)
    else:
        # create one fake texture
        # noise = netG.generate_noise(batch_size=1).to(device)
        # for now nz = 512, 256 values comes from randn and other 256 from encoded uv layout
        noise = torch.randn(nz // 2).to(device) if use_uv_layout_encoder else netG.generate_noise(batch_size=1).to(
            device)

        if use_uv_layout_encoder:
            # load uv_layout image
            uv_image = Image.open(uv_layout_path).convert('RGB')
            transform = transforms.ToTensor()
            uv_tensor = transform(uv_image).unsqueeze_(dim=0).to(device)
            uv_z = encoder(uv_tensor)

            # concatenate noise and uv_z and reshape
            noise = torch.cat([noise, uv_z.squeeze(dim=0)]).reshape(batch_size, -1, 1, 1).to(device=device)

        generated_images = netG(noise)

    # texture = netG(noise).permute(0, 2, 3, 1)
    texture = generated_images.permute(0, 2, 3, 1)

    # apply fake texture to given mesh with previous configuration
    mesh.textures = TexturesUV(texture.to(device), faces_uvs=faces_uvs, verts_uvs=verts_uvs)

    # will produce fake images of shape [batch_sizeximage_sizeximage_sizex4(RGBA)]
    fake_images = renderer(mesh)
    fake_batch = fake_images[..., :3].to(device).permute(0, 3, 1, 2)

    if debug:
        plot_original_and_generated(rendered_images, fake_images.detach())

    label.fill_(fake_label)
    # pass fake batch to netD
    if stylegan2:
        fake_pred = netD(fake_batch.detach())
        d_loss = d_logistic_loss(real_pred, fake_pred)
        d_loss.backward()
    else:
        output = netD(fake_batch.detach()).view(-1)
        errD_fake = criterion(output, label)
        # calculated the gradient of this fake batch: gradients will accumulate with previous real_batch
        errD_fake.backward()

    # update the discriminator
    optimizerD.step()

    # list some values for plot
    if stylegan2:
        errD = d_loss.item()
    else:
        errD = errD_real.item() + errD_fake.item()

    loss_D.append(errD)

    # Train generator
    # zero grad the generator
    netG.zero_grad()

    if use_uv_layout_encoder and not stylegan2:
        # zero grad the Encoder
        encoder.zero_grad()

    label.fill_(real_label)
    # D got updated, make another pass to create computation graph and then to calculate the gradients
    if stylegan2:
        fake_pred = netD(fake_batch)
        errG = g_nonsaturating_loss(fake_pred)
    else:
        output = netD(fake_batch).view(-1)
        # calculate loss: gan loss
        errG = criterion(output, label)

    # calculate gradients
    if gan_only:
        errG.backward()
    else:
        # calculate image loss
        image_loss = torch.mean(torch.abs(fake_images[0, ...] - rendered_images[0, ...]))
        loss = l1_weight * image_loss + gan_weight * errG
        loss.backward()

        loss_I.append(image_loss.item())

    # update generator
    optimizerG.step()

    if use_uv_layout_encoder:
        # update encoder
        optimizerE.step()

    # some book keeping
    loss_G.append(errG.item())

    # Output training stats
    if gan_only:
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch + 1, num_epochs, errD, errG.item()))
    else:
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tLoss_I: %.4f' % (
        epoch + 1, num_epochs, errD, errG.item(), image_loss.item()))

# plot losses
plt.figure(figsize=(10, 5))
plt.title("Loss During Training")
plt.plot(loss_G, label="G")
plt.plot(loss_D, label="D")
if not gan_only:
    plt.plot(loss_I, label="Image")

plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# save the trained generator
torch.save(netG.state_dict(), save_path_generator)

if use_uv_layout_encoder and not stylegan2:
    # save the trained encoder
    torch.save(encoder.state_dict(), save_path_encoder)

if is_shapenet:
    shapenet_model = shapenet_dataset[0]
    synset_id = shapenet_model["synset_id"]
    model_id = shapenet_model["model_id"]
    if mesh_approximation:
        obj_path = config["approx_mesh_path"]
    else:
        obj_path = shapenet_path + "/" + synset_id + "/" + model_id + "/models/" + "bpy_model.obj"
    uv_layout_path = shapenet_path + "/" + synset_id + "/" + model_id + "/models/uv_layout.png" if use_uv_layout_encoder else None

# generate one texture image
with torch.no_grad():
    if stylegan2:
        get_latents_fn = mixed_list if random() < mixed_prob else noise_list
        style = get_latents_fn(n=batch_size, layers=num_layers, latent_dim=latent_dim, device=device)
        noise = image_noise(n=batch_size, im_size=image_size, device=device)

        w_space = latent_to_w(netS, style)
        w_styles = styles_def_to_tensor(w_space)

        generated_images = netG(w_styles, noise)
    else:
        # create one fake texture
        # noise = netG.generate_noise(batch_size=1).to(device)
        # for now nz = 512, 256 values comes from randn and other 256 from encoded uv layout
        noise = torch.randn(nz // 2).to(device) if use_uv_layout_encoder else netG.generate_noise(batch_size=1).to(
            device)

        if use_uv_layout_encoder:
            # load uv_layout image
            uv_image = Image.open(uv_layout_path).convert('RGB')
            transform = transforms.ToTensor()
            uv_tensor = transform(uv_image).unsqueeze_(dim=0).to(device)
            uv_z = encoder(uv_tensor)

            # concatenate noise and uv_z and reshape
            noise = torch.cat([noise, uv_z.squeeze(dim=0)]).reshape(batch_size, -1, 1, 1).to(device=device)

        generated_images = netG(noise)

    # texture = netG(noise).permute(0, 2, 3, 1)
    texture = generated_images.permute(0, 2, 3, 1)

plt.imshow(texture.squeeze().cpu().numpy())
plt.title("sample texture")
plt.show()

if is_shapenet:
    verts, faces, aux = load_obj(obj_path, device=device)
    mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])
    # load the mesh with uv information and apply generated texture
    faces_uvs = [faces.textures_idx]
    # verts_uvs = mesh.textures._verts_uvs_list
    verts_uvs = [aux.verts_uvs]
else:
    mesh = load_objs_as_meshes([obj_path], device=device)
    faces_uvs = mesh.textures.faces_uvs_list()
    verts_uvs = mesh.textures.verts_uvs_list()

if not is_shapenet:
    # Normalize
    verts = mesh.verts_packed()
    N = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center.expand(N, 3))
    mesh.scale_verts_((1.0 / float(scale)))

distance = 1
elevation = 15.0
azimuth = 90.0

R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation, azimuth=azimuth,
                                                        degrees=True, device=str(device))

cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

if is_shapenet:
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                            cull_backfaces=True)
else:
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                            cull_backfaces=False)

# Place a point light in front of the object: put point light at the location of camera
lights = PointLights(location=camera_location, device=str(device))

# Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
# interpolate the texture uv coordinates for each vertex, sample from a texture image and
# apply the Phong lighting model
renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                        shader=HardFlatShader(cameras=cameras, lights=lights, device=device) if is_shapenet else
                        SoftPhongShader(device=device, cameras=cameras, lights=lights))

if is_shapenet:
    original = shapenet_dataset.render(model_ids=[model_id], device=device, cameras=cameras,
                                       raster_settings=raster_settings,
                                       lights=lights, shader_type=HardFlatShader)
else:
    original = renderer(mesh, cameras=cameras, lights=lights)

mesh.textures = TexturesUV(texture.to(device), faces_uvs=faces_uvs, verts_uvs=verts_uvs)
# render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
image = renderer(mesh, cameras=cameras, lights=lights)
plt.figure(figsize=(7, 7))

plt.subplot(1, 2, 1)
plt.imshow(original[0, ..., :3].cpu().numpy())
plt.title("Rendered with original texture")
plt.grid("off")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image[0, ..., :3].cpu().numpy())
plt.title("Rendered with synthetic texture")
plt.grid("off")
plt.axis("off")
plt.show()
