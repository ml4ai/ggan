"""
test if the texture generated for one model can be used for another model
"""
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturesUV
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer import HardFlatShader
import warnings
from PIL import Image
from torchvision import transforms
from pytorch3d.datasets import ShapeNetCore
warnings.filterwarnings("ignore")

# device to train: gpu/cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# transfer the texture of src_index to mesh of target_index
# target_index: shapenet index for the reference mesh
target_index = 11
# src_index: shapenet index for the source mesh
src_index = 21

# load target mesh
obj_path = "/home/user/data/realsim/final_model_" + str(target_index) + ".obj"
verts, faces, aux = load_obj(obj_path, device=device)
mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])

# load the mesh with uv information and apply generated texture
icosphere_path = "/home/user/data/realsim/icosphere.obj"
verts_temp, faces_temp, aux_temp = load_obj(icosphere_path, device=device)
faces_uvs = [faces_temp.textures_idx]
# verts_uvs = mesh.textures._verts_uvs_list
verts_uvs = [aux_temp.verts_uvs]

# load the saved texture
texture_path = "/home/user/data/realsim/texture_" + str(src_index) + "_cropped.png"
transform = transforms.ToTensor()
texture = transform(Image.open(texture_path))
texture = texture.permute(1, 2, 0).unsqueeze_(dim=0)[..., :3]

# load shapenet model for original rendering
SHAPENET_PATH = "/home/user/data/car_full"
shapenet_dataset = ShapeNetCore(SHAPENET_PATH, synsets=["car"], version=2, load_textures=True, texture_resolution=4)

shapenet_model = shapenet_dataset[target_index]
synset_id = shapenet_model["synset_id"]
model_id = shapenet_model["model_id"]

distance = 1
elevation = 15.0
azimuth = 90.0

R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation, azimuth=azimuth,
                                                        degrees=True, device=str(device))

cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1, cull_backfaces=True)

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
