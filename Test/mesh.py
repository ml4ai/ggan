"""
test our model using spherical uv mapping instead of different uv map per mesh
loads the saved generator and generates texture
"""
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturesUV
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer import HardFlatShader
from pytorch3d.datasets import ShapeNetCore
import warnings
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

# in case of shapenet - specify category
category = "car"

# save path for trained generator
save_path_generator = "Models/generator.pth"

# spatial size of image generated from generator/ input to Discriminator
image_size = 512

if image_size == 64:
    from Generator.generator64 import Generator
elif image_size == 128:
    from Generator.generator128 import Generator
elif image_size == 512:
    from Generator.generator512 import Generator
else:
    from Generator.generator1024 import Generator

# load weights from saved file for generator
netG = Generator(args.configG)
netG.load_state_dict(torch.load(save_path_generator))
netG = netG.to(device)
netG.eval()

# shapenet test index
test_index = 11

SHAPENET_PATH = "/home/user/data/car_full"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, synsets=["car"], version=2, load_textures=True, texture_resolution=4)
approx_obj_path = "/home/user/data/final_model_11.obj"

shapenet_model = shapenet_dataset[test_index]
synset_id = shapenet_model["synset_id"]
model_id = shapenet_model["model_id"]

verts, faces, aux = load_obj(approx_obj_path, device=device)
mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])

# load the mesh with uv information and apply generated texture
icosphere_path = "/home/user/data/icosphere.obj"
verts_temp, faces_temp, aux_temp = load_obj(icosphere_path, device=device)
faces_uvs = [faces_temp.textures_idx]
# verts_uvs = mesh.textures._verts_uvs_list
verts_uvs = [aux_temp.verts_uvs]

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
