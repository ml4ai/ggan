"""
Train code for shapenet car dataset. Loads the 3d mesh with texture using pytorch3d, renders them as real image, applies
generated texture to the same mesh and renders them as fake images: trains our whole pipeline on shapenet car dataset
"""

import argparse
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_obj
from Utils.utils import weights_init
from torch import nn
from torch import optim
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer import TexturesUV, RasterizationSettings
from pytorch3d.structures import Meshes
import time
import random
# from Utils.utils import get_gpu_memory_map
# from Utils.plot import plot_gpu_usage
from Utils.plot import plot_losses, plot_original_and_generated, plot_rendered_images
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.renderer import HardFlatShader
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer import MeshRenderer, MeshRasterizer
import warnings
# from PIL import UnidentifiedImageError
from Models.encoder import Encoder
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

# Analysze time it takes to run the code and memory used
# with clean up inside loop and without it
start_time = time.time()

# some clean up
clean = True

# save path for trained generator
save_path_generator = "Models/generator.pth"
save_path_encoder = "Models/encoder.pth"

# debug: set it to false before training, used to visualize the rendered images
debug = False

# batch size used for training
batch_size = 1

# train with multiple losses: gan loss, l1 loss between image rendered with synthetic texture and rendered with original texture
# weight for l1 loss: weight for l1 loss
l1_weight = 0.5
# weight for gan loss: weight for gan loss
gan_weight = 0.5

# set gan_only just to train from gan losses
gan_only = True

# spatial size of image generated from generator/ input to Discriminator
image_size = 512

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

# num epochs to train
num_epochs = 500

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# device to train: gpu/cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create a list of obj files with single texture that we can later iterate over
# SHAPENET_PATH = "/home/user/data/ShapeNetCore.v2"
SHAPENET_PATH = "/home/user/data/car_full/"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, synsets=["car"], version=2, load_textures=True, texture_resolution=4)

# parse the arguments for config of Generator and config of Discriminator
parser = argparse.ArgumentParser()
parser.add_argument("--configG", help="configuration file for Generator")
parser.add_argument("--configD", help="configuration file for Discriminator")
args = parser.parse_args()

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

# encoder
encoder = Encoder(in_channels=3, out_dim=512).to(device)

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

# optimizer for Discriminator
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizer for Generator
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
# optimizer for Encoder
optimizerE = optim.Adam(encoder.parameters(), lr=lr, betas=(beta1, 0.999))

loss_G = []
loss_D = []

print("Training...")
# gpu_usage = []

# for now train on 10 objects and test on 1 object: to make sure the code can learn texture from multiple types of cars
# and apply it to unseen model

train_indices = [11, 21, 27, 77, 89, 102, 146, 150, 169, 181]
test_indices = [187]

# total_obj = len(shapenet_dataset)
total_obj = len(train_indices)

for epoch in range(num_epochs):
    # for each epoch iterate over the obj files and train
    print(f"Epoch: {epoch + 1} / {num_epochs}")
    obj_num = 1

    """
    uncomment to train on all objects
    for idx in range(total_obj):
        # load the blender modified version of the same model by load_obj method
        # As it has the uv mapping information while the original obj file doesn't have it
        try:
            shapenet_model = shapenet_dataset[idx]
        # Some objects can't be loaded because of IndexError while creating TextureAtlas: [seems to be an issue with pytorch3d]
        # but this happens for only few objects, so we are fine for now: if we can't load properly, we go to next obj
        # some texture images are in .psd format and they throw PIL.UnidentifiedImageError
        except (IndexError, UnidentifiedImageError) as e:
            continue
    """
    for idx in train_indices:
        shapenet_model = shapenet_dataset[idx]

        synset_id = shapenet_model["synset_id"]
        model_id = shapenet_model["model_id"]
        obj_filename = SHAPENET_PATH + "/" + synset_id + "/" + model_id + "/models/" + "bpy_model.obj"
        uv_layout_path = SHAPENET_PATH + "/" + synset_id + "/" + model_id + "/models/uv_layout.png"

        verts, faces, aux = load_obj(obj_filename, device=device)
        mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])
        # load the mesh with uv information and apply generated texture
        faces_uvs = [faces.textures_idx]
        # verts_uvs = mesh.textures._verts_uvs_list
        verts_uvs = [aux.verts_uvs]

        # sample new camera location
        distance = round(random.uniform(1.0, 2.0), 2)
        # generate batch of meshes and rander them: real
        meshes = mesh.extend(batch_size)

        # Randomization: sample batch_size*2 values for each, randomly permute them and select the batch_size values
        elevation = torch.FloatTensor(batch_size).uniform_(0, 180)
        azimuth = torch.FloatTensor(batch_size).uniform_(-180, 180)

        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation,
                                                                azimuth=azimuth,
                                                                degrees=True, device=str(device))

        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

        # render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
        # use shapenet renderer: that can render with loaded TextureAtlas
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                cull_backfaces=True)

        # Place a point light in front of the object: for now let's put at the same location of camera:
        # so the rendered image will get some
        # light, we can also place them at s*camera_location where s is some scalar
        lights = PointLights(location=camera_location, ambient_color=((0.5, 0.5, 0.5),),
                             diffuse_color=((0.4, 0.4, 0.4),),
                             specular_color=((0.1, 0.1, 0.1),), device=str(device))

        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                                shader=HardFlatShader(cameras=cameras, lights=lights, device=device))

        images = shapenet_dataset.render(model_ids=[model_id], device=device, cameras=cameras,
                                         raster_settings=raster_settings,
                                         lights=lights, shader_type=HardFlatShader)

        if debug:
            plot_rendered_images(images)

        # zero grad the netD
        netD.zero_grad()

        # extract RGB images from these real rendered images
        real_batch = images[..., :3].to(device).permute(0, 3, 1, 2)
        label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
        # forward pass the real batch through D
        output = netD(real_batch).view(-1)
        errD_real = criterion(output, label)
        # calculate gradients for real batch
        errD_real.backward()

        # create one fake texture
        # noise = netG.generate_noise(batch_size=1).to(device)
        # for now nz = 200, 100 values comes from randn and other 100 from encoded uv layout
        noise = torch.randn(512).to(device)

        # load uv_layout image
        uv_image = Image.open(uv_layout_path).convert('RGB')
        transform = transforms.ToTensor()
        uv_tensor = transform(uv_image).unsqueeze_(dim=0).to(device)
        uv_z = encoder(uv_tensor)

        # concatenate noise and uv_z and reshape
        # noise = torch.cat([noise, uv_z.squeeze(dim=0)]).reshape(batch_size, -1, 1, 1).to(device=device)
        noise = (noise + uv_z.squeeze(dim=0)).reshape(batch_size, -1, 1, 1).to(device)

        texture = netG(noise).permute(0, 2, 3, 1)
        # apply fake texture to given mesh with previous configuration
        mesh.textures = TexturesUV(texture.to(device), faces_uvs=faces_uvs, verts_uvs=verts_uvs)

        # will produce fake images of shape [batch_sizeximage_sizeximage_sizex4(RGBA)]
        fake_images = renderer(mesh)
        fake_batch = fake_images[..., :3].to(device).permute(0, 3, 1, 2)
        label.fill_(fake_label)
        # pass fake batch to netD
        output = netD(fake_batch.detach()).view(-1)
        errD_fake = criterion(output, label)
        # calculated the gradient of this fake batch: gradients will accumulate with previous real_batch
        errD_fake.backward()
        # update the discriminator
        optimizerD.step()

        # list some values for plot
        errD = errD_real.item() + errD_fake.item()
        loss_D.append(errD)

        # Train generator
        # zero grad the generator
        netG.zero_grad()

        # zero grad the Encoder
        encoder.zero_grad()

        label.fill_(real_label)
        # D got updated, make another pass to create computation graph and then to calculate the gradients
        output = netD(fake_batch).view(-1)
        # calculate loss
        errG = criterion(output, label)
        # calculate gradients
        if gan_only:
            errG.backward()
        else:
            # calculate image loss
            image_loss = torch.mean(torch.abs(fake_images[0, ...] - images[0, ...]))
            loss = l1_weight * image_loss + gan_weight * errG
            loss.backward()

        # update generator
        optimizerG.step()

        # update encoder
        optimizerE.step()

        # some book keeping
        loss_G.append(errG.item())

        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (obj_num, total_obj, errD, errG.item()))
        obj_num += 1

        # clean up
        if clean:
            del mesh, cameras, renderer, verts_uvs, faces_uvs, images, fake_images, texture

        # gpu_usage.append(get_gpu_memory_map()[0])

# save the trained generator
torch.save(netG.state_dict(), save_path_generator)
# save the trained encoder
torch.save(encoder.state_dict(), save_path_encoder)

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print(f"Time: {round(elapsed_minutes, 2)} Minutes")

# plot losses
plot_losses(loss_G, loss_D)

# plot gpu_usage
# plot_gpu_usage(gpu_usage=gpu_usage)

shapenet_model = shapenet_dataset[test_indices[0]]
synset_id = shapenet_model["synset_id"]
model_id = shapenet_model["model_id"]
obj_filename = SHAPENET_PATH + "/" + synset_id + "/" + model_id + "/models/" + "bpy_model.obj"
uv_layout_path = SHAPENET_PATH + "/" + synset_id + "/" + model_id + "/models/uv_layout.png"

# generate one texture image
with torch.no_grad():
    # noise = netG.generate_noise(batch_size=1).to(device)
    # for now nz = 200, 100 values comes from randn and other 100 from encoded uv layout
    noise = torch.randn(512).to(device)

    # load uv_layout image
    uv_image = Image.open(uv_layout_path).convert('RGB')
    transform = transforms.ToTensor()
    uv_tensor = transform(uv_image).unsqueeze_(dim=0).to(device)
    uv_z = encoder(uv_tensor)

    # concatenate noise and uv_z and reshape
    # noise = torch.cat([noise, uv_z.squeeze(dim=0)]).reshape(batch_size, -1, 1, 1).to(device=device)
    noise = (noise + uv_z.squeeze(dim=0)).reshape(batch_size, -1, 1, 1).to(device)

    texture = netG(noise).permute(0, 2, 3, 1)

plt.imshow(texture.squeeze().cpu().numpy())
plt.title("sample texture")
plt.show()

verts, faces, aux = load_obj(obj_filename, device=device)
mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])
# load the mesh with uv information and apply generated texture
faces_uvs = [faces.textures_idx]
# verts_uvs = mesh.textures._verts_uvs_list
verts_uvs = [aux.verts_uvs]

distance = 1.0
elevation = 15.0
azimuth = 120.0

R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation, azimuth=azimuth,
                                                        degrees=True, device=str(device))

cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

# original = renderer(mesh, cameras=cameras, lights=lights)
raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True)

# Place a point light in front of the object: put point light at the location of camera
lights = PointLights(location=camera_location, device=str(device))

renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                        shader=HardFlatShader(cameras=cameras, lights=lights, device=device))

original = shapenet_dataset.render(model_ids=[model_id], device=device, cameras=cameras, raster_settings=raster_settings,
                                   lights=lights, shader_type=HardFlatShader)

mesh.textures = TexturesUV(texture.to(device), faces_uvs=faces_uvs, verts_uvs=verts_uvs)
# render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
image = renderer(mesh, cameras=cameras, lights=lights)

plot_original_and_generated(original, image)
