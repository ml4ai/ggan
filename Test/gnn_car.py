"""
Test the trained GNN for generating texture
Download the model from kraken and use it to generate new textures for a given model
"""
import matplotlib.pyplot as plt
import argparse
import json
import torch
from Models.gcn import GCN
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from Utils.mesh_utils import mesh_features, mesh_features_dual
from pytorch3d.renderer import TexturesVertex, look_at_view_transform, camera_position_from_spherical_angles
from pytorch3d.renderer.cameras import FoVPerspectiveCameras
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, RasterizationSettings
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer import HardFlatShader, TexturesAtlas
import pickle
import warnings

warnings.filterwarnings("ignore")

# parse the arguments for config of Generator and config of Discriminator
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration file for paths and hyperparameters")
# we will use GCN as the generator
# parser.add_argument("--configG", help="configuration file for Generator")
parser.add_argument("--configD", help="configuration file for Discriminator")
parser.add_argument("--model_idx", default=485, type=int, help="model index [485:500]")
parser.add_argument("--model_path", type=str, default="/home/kcdharma/results/arete-realsim/gnn.pth",
                    help="path to trained model")
parser.add_argument("--type", type=str, default="vertex", help="vertex/dual")
parser.add_argument("--out_res", type=int, default=1, help="output resolution of TextureAtlas")

args = parser.parse_args()

config = json.load(open(args.config, 'r'))
# parse config file and get the parameters

# load train_ids from model_ids.pickle file
model_ids_path = "Data/model_ids.pickle"
with open(model_ids_path, "rb") as read_file:
    model_ids = pickle.load(read_file)

model_ids = sorted(model_ids)
# output resolution of texture atlas
out_res = int(args.out_res)
train_type = args.type
if train_type == "dual":
    # 18 for now [x, y, z] for 3 vertices and [xn, yn, zn] for 3 vertices
    in_features = config["dual_in_features"]
    out_features = out_res * out_res * 3
else:
    # in_features 3 [x, y, z] for each vertex or 6 [x, y, z, normal_x, normal_y, normal_z]
    # for each vertex
    in_features = config["in_features"]
    # out_features: 3 [r, g, b] color for each vertex
    out_features = config["out_features"]

# number of hidden features transforms position/normal information ---> latent space
n_hidden = config["n_hidden"]
# gpu_index to run the inference
gpu_index = config["gpu_index"]
# batch size used for training
# batch_size = config["batch_size"]
batch_size = 1
# device to train: gpu/cpu
device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")

# path to trained gnn [Downloaded from kraken]
model_path = args.model_path

# in case of shapenet - specify category
category = config["category"]

# spatial size of image generated from generator/ input to Discriminator / rendered
image_size = config["image_size"]
noise_dim = config["noise_dim"]

# netG and netD
print("Initializing Generator...")
netG = GCN(in_features=in_features, n_hidden=n_hidden, out_features=out_features,
           noise_dim=noise_dim, dropout=0.1).to(device)

if torch.cuda.is_available():
    netG.load_state_dict(torch.load(model_path))
else:
    netG.load_state_dict(torch.load(model_path, map_location="cpu"))

# set it to eval mode [no dropout]
netG.eval()
print(netG)

# if shapenet: provide the shapenet paths
shapenet_path = config["shapenet_path"]
shapenet_dataset = ShapeNetCore(shapenet_path, synsets=[category], version=2, load_textures=True,
                                texture_resolution=4)
# shapenet index
model_index = int(args.model_idx)
synset_id = "02958343"
model_id = model_ids[model_index]
# load obj file
obj_path = shapenet_path + "/" + synset_id + "/" + model_id + "/models/" + "model_normalized.obj"

verts, faces, aux = load_obj(obj_path, device=device)
mesh = Meshes([verts.to(device)], [faces.verts_idx.to(device)])

if train_type == "dual":
    features, adj = mesh_features_dual(obj_path=obj_path)
else:
    # calculate features and adj
    features, adj = mesh_features(obj_path=obj_path)

features, adj = features.to(device), adj.to(device)

print("Generating texture...")
for _ in range(1):
    distance = 1.0
    elevation = torch.FloatTensor(batch_size).uniform_(0, 15)
    azimuth = torch.FloatTensor(batch_size).uniform_(30, 90)

    R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
    camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation, azimuth=azimuth,
                                                            degrees=True, device=str(device))

    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                            cull_backfaces=True)

    # Place a point light in front of the object: put point light at the location of camera
    lights = PointLights(location=camera_location, device=str(device))

    # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                            shader=HardFlatShader(cameras=cameras, lights=lights, device=device)).to(device)

    # rendere with original texture
    original = shapenet_dataset.render(model_ids=[model_id], device=device, cameras=cameras,
                                       raster_settings=raster_settings,
                                       lights=lights, shader_type=HardFlatShader)

    # generate one texture image and render with synthetic texture
    with torch.no_grad():
        noise = torch.randn(noise_dim).to(device)
        texture = netG(features, adj, noise).to(device).unsqueeze(dim=0)
        if train_type == "dual":
            texture = torch.reshape(texture, (texture.shape[0],
                                    texture.shape[1],
                                    out_res, out_res,
                                    3)).to(device)
            mesh.textures = TexturesAtlas(texture)
        else:
            mesh.textures = TexturesVertex(verts_features=texture)

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
