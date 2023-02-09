# load the trained generator
# test on multiple test images and show the results

import torch
import matplotlib.pyplot as plt
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform
from pytorch3d.renderer import FoVPerspectiveCameras, TexturesUV
from Utils.utils import normalize_mesh, get_lights, get_renderer
from Utils.plot import plot_original_and_generated, visualize_3d
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument("--configG", help="configuration file for Generator")
parser.add_argument("--configD", help="configuration file for Discriminator")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

split_path = "Models/split.pickle"

with open(split_path, "rb") as handle:
    split = pickle.load(handle)

train_files = split["train_files"]
test_files = split["test_files"]

print(f"train_files: {len(train_files)}")
print(f"test_files: {len(test_files)}")

# test_index = torch.randint(0, len(test_files), (1,)).item()
test_index = 6
obj_filename = test_files[test_index]

# generate one texture image
with torch.no_grad():
    noise = netG.generate_noise(batch_size=1).to(device)
    texture = netG(noise).permute(0, 2, 3, 1)

plt.imshow(texture.squeeze().cpu().numpy())
plt.title("sample texture")
plt.show()

mesh = load_objs_as_meshes([obj_filename], device=device)

mesh = normalize_mesh(mesh)

faces_uvs = mesh.textures.faces_uvs_list()
verts_uvs = mesh.textures.verts_uvs_list()

R, T = look_at_view_transform(dist=2.0, elev=0, azim=30)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
# get lights
lights = get_lights(device=device)
# get renderer
renderer = get_renderer(image_size=image_size, cameras=cameras, lights=lights, device=device)

original = renderer(mesh, cameras=cameras, lights=lights)
mesh.textures = TexturesUV(texture.to(device), faces_uvs=faces_uvs, verts_uvs=verts_uvs)
# render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
image = renderer(mesh, cameras=cameras, lights=lights)

plot_original_and_generated(original, image)

visualize_3d(mesh)
