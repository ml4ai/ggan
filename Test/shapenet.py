"""
Test code for shapenet dataset: will load saved generator, generate a texture from latent noise and apply that
texture to one 3d mesh from shapenet dataset [car category: can be changed] and renders the image as well as open
the 3d visualization of textured mesh in default browser using plotly and pytorch3d
"""

import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_obj
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer import TexturesUV, RasterizationSettings
from pytorch3d.structures import Meshes
from Utils.plot import plot_original_and_generated
from pytorch3d.datasets import ShapeNetCore
import argparse
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer import MeshRasterizer, MeshRenderer, HardFlatShader
import warnings
from Models.encoder import Encoder
from PIL import Image
from torchvision import transforms

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--configG", help="configuration file for Generator")
parser.add_argument("--configD", help="configuration file for Discriminator")
args = parser.parse_args()

# test index to test on shapenet car objs
test_index = 11

# save path for trained generator
save_path_generator = "Models/generator.pth"
save_path_encoder = "Models/encoder.pth"

# spatial size of image generated from generator/ input to Discriminator
image_size = 512
# device to train: gpu/cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

# load the weights from saved encoder
encoder = Encoder(in_channels=3, out_dim=512)
encoder.load_state_dict(torch.load(save_path_encoder))
encoder = encoder.to(device)
encoder.eval()

SHAPENET_PATH = "/home/user/data/car_full"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, synsets=["car"], version=2, load_textures=True, texture_resolution=4)

shapenet_model = shapenet_dataset[test_index]
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
    # noise = torch.cat([noise, uv_z.squeeze(dim=0)]).reshape(1, -1, 1, 1).to(device=device)
    noise = (noise + uv_z.squeeze(dim=0)).reshape(1, -1, 1, 1).to(device)

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
