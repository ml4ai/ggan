"""
train our model using spherical uv mapping instead of different uv map per mesh
approximate the mesh first deform.py and and then provide the path to approximated mesh and the target meshes
spherical uv mapping has the advantage: as the mapping from vertices to texture space will remain constant and
the textures learned from one model can be applied to another 3d model
"""
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from Utils.utils import weights_init
from torch import nn
from torch import optim
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturesUV
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer import HardFlatShader
from pytorch3d.datasets import ShapeNetCore
import warnings
from Utils.plot import plot_original_and_generated
import argparse

warnings.filterwarnings("ignore")

# parse the arguments for config of Generator and config of Discriminator
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration file for paths and hyperparameters")
parser.add_argument("--configG", help="configuration file for Generator")
parser.add_argument("--configD", help="configuration file for Discriminator")
args = parser.parse_args()

# device to train: gpu/cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# batch size used for training
batch_size = 1
# in case of shapenet - specify category
category = "car"

# save path for trained generator
save_path_generator = "Models/generator.pth"

# if debug: will show the image rendered with original texture and rendered with synthetic texture during training
debug = False

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

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

# num epochs to train
num_epochs = 1000

# if shapenet: provide the shapenet paths
shapenet_path = "/home/user/data/car_full"
shapenet_dataset = ShapeNetCore(shapenet_path, synsets=[category], version=2, load_textures=True, texture_resolution=4)

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

# loss for generator
loss_G = []
# loss for discriminator
loss_D = []

# shapenet index
model_index = 21
shapenet_model = shapenet_dataset[model_index]
# get some information
model_id = shapenet_model["model_id"]

# load obj file
obj_path = "/home/user/data/final_model_" + str(model_index) + ".obj"

verts, faces, aux = load_obj(obj_path, device=device)
mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])

# load the mesh with uv information and apply generated texture
icosphere_path = "/home/user/data/icosphere.obj"
verts_temp, faces_temp, aux_temp = load_obj(icosphere_path, device=device)
faces_uvs = [faces_temp.textures_idx]
# verts_uvs = mesh.textures._verts_uvs_list
verts_uvs = [aux_temp.verts_uvs]

print("Training...")
for epoch in range(num_epochs):
    # for each epoch iterate over the obj files and train
    print(f"Epoch: {epoch + 1} / {num_epochs}")

    # sample new camera location
    distance = 1.0
    elevation = torch.FloatTensor(batch_size).uniform_(-10, 180)
    azimuth = torch.FloatTensor(batch_size).uniform_(-180, 180)

    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation, azimuth=azimuth,
                                                            degrees=True, device=str(device))

    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True)

    # Place a point light in front of the object: for now let's put at the same location of camera: so the rendered image will get some
    # light, we can also place them at s*camera_location where s is some scalar
    lights = PointLights(location=camera_location, ambient_color=((0.5, 0.5, 0.5),), diffuse_color=((0.4, 0.4, 0.4),),
                         specular_color=((0.1, 0.1, 0.1),), device=str(device))

    # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                            shader=HardFlatShader(cameras=cameras, lights=lights, device=device))

    # render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
    rendered_images = shapenet_dataset.render(model_ids=[model_id], device=device, cameras=cameras, raster_settings=raster_settings,
                                              lights=lights, shader_type=HardFlatShader)

    # zero grad the netD
    netD.zero_grad()
    # extract RGB images from these real rendered images
    real_batch = rendered_images[..., :3].to(device).permute(0, 3, 1, 2)
    label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
    # forward pass the real batch through D
    output = netD(real_batch).view(-1)
    errD_real = criterion(output, label)
    # calculate gradients for real batch
    errD_real.backward()

    # create one fake texture
    # noise = netG.generate_noise(batch_size=1).to(device)
    # for now nz = 512, 256 values comes from randn and other 256 from encoded uv layout
    noise = netG.generate_noise(batch_size=1).to(device)
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

    label.fill_(real_label)
    # D got updated, make another pass to create computation graph and then to calculate the gradients
    output = netD(fake_batch).view(-1)
    # calculate loss: gan loss
    errG = criterion(output, label)

    # calculate gradients
    errG.backward()

    # update generator
    optimizerG.step()

    # some book keeping
    loss_G.append(errG.item())

    # Output training stats
    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch + 1, num_epochs, errD, errG.item()))

# plot losses
plt.figure(figsize=(10, 5))
plt.title("Loss During Training")
plt.plot(loss_G, label="G")
plt.plot(loss_D, label="D")

plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

# save the trained generator
torch.save(netG.state_dict(), save_path_generator)

# generate one texture image
with torch.no_grad():
    # create one fake texture
    noise = netG.generate_noise(batch_size=1).to(device)
    generated_images = netG(noise)
    # texture = netG(noise).permute(0, 2, 3, 1)
    texture = generated_images.permute(0, 2, 3, 1)

plt.imshow(texture.squeeze().cpu().numpy())
plt.title("sample texture")
plt.show()

distance = 1
elevation = 15.0
azimuth = 90.0

R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation, azimuth=azimuth,
                                                        degrees=True, device=str(device))

cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True)

# Place a point light in front of the object: put point light at the location of camera
lights = PointLights(location=camera_location, device=str(device))

# Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
# interpolate the texture uv coordinates for each vertex, sample from a texture image and
# apply the Phong lighting model
renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                        shader=HardFlatShader(cameras=cameras, lights=lights, device=device))

original = shapenet_dataset.render(model_ids=[model_id], device=device, cameras=cameras, raster_settings=raster_settings,
                                   lights=lights, shader_type=HardFlatShader)

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
