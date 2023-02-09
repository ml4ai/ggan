# This one performs latent interpolation for real world things dataset

import argparse
import json
import torch
import numpy as np
import pickle
from pytorch3d.io import load_objs_as_meshes
from Utils.utils import normalize_mesh, get_lights, get_renderer
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, TexturesUV
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# parse the arguments for config of Generator and config of Discriminator
parser = argparse.ArgumentParser()
parser.add_argument("--configG", help="configuration file for Generator")
args = parser.parse_args()

config_file = args.configG
config = json.load(open(config_file, "r"))
interpolation_type = config["interpolation_type"]
nz = config["nz"]
num_interpolation = config["num_interpolation"]

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

save_path = "Models/generator.pth"

# load weights from saved file for generator
netG = Generator(args.configG)
netG.load_state_dict(torch.load(save_path))
netG = netG.to(device)
netG.eval()

vectors = []
# generate two random vectors
p1 = torch.randn(1, nz, 1, 1).to(device)
p2 = torch.randn(1, nz, 1, 1).to(device)

if interpolation_type == "uniform":
    alphas = torch.linspace(0, 1, num_interpolation)
    for alpha in alphas:
        vector = (1 - alpha) * p1 + alpha * p2
        vectors.append(vector)


# Ref: https://github.com/soumith/dcgan.torch/issues/14

def slerp(val, low, high):
    omega = np.arccos(np.clip(np.dot(low / np.linalg.norm(low), high / np.linalg.norm(high)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - val) * low + val * high  # L'Hopital's rule/LERP
    return np.sin((1.0 - val) * omega) / so * low + np.sin(val * omega) / so * high


if interpolation_type == "spherical":
    alphas = torch.linspace(0, 1, num_interpolation)
    for alpha in alphas:
        vector = slerp(alpha, p1, p2)
        vectors.append(torch.from_numpy(vector))

split_path = "Models/split.pickle"

with open(split_path, "rb") as handle:
    split = pickle.load(handle)

train_files = split["train_files"]
test_files = split["test_files"]

test_index = torch.randint(0, len(test_files), (1,)).item()
obj_filename = test_files[test_index]
# apply generated texture to given mesh and render the image
mesh = load_objs_as_meshes([obj_filename], device=device)
mesh = normalize_mesh(mesh)

faces_uvs = mesh.textures.faces_uvs_list()
verts_uvs = mesh.textures.verts_uvs_list()

R, T = look_at_view_transform(dist=2.0, elev=0, azim=90)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
# get lights
lights = get_lights(device=device)
# get renderer
renderer = get_renderer(image_size=image_size, cameras=cameras, lights=lights, device=device)

original = renderer(mesh, cameras=cameras, lights=lights)

images = []
for vector in vectors:
    texture = netG(vector).permute(0, 2, 3, 1).to(device)
    mesh.textures = TexturesUV(texture.to(device), faces_uvs=faces_uvs, verts_uvs=verts_uvs)
    # render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
    image = renderer(mesh, cameras=cameras, lights=lights)
    images.append(image)

total = len(images)
for img in images:
    for i in range(total):
        plt.subplot(1, total, 1 + i)
        plt.axis("off")
        plt.imshow(img[0, ..., :3].detach().cpu().numpy())

plt.show()
