"""
Train on real world textured things dataset
available here: https://texturedmesh.isti.cnr.it/
"""

import argparse
import matplotlib.pyplot as plt
import torch
from pytorch3d.io import load_objs_as_meshes
from Utils.utils import weights_init
from torch import nn
from torch import optim
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer import TexturesUV
import time
from Utils.utils import get_gpu_memory_map, normalize_mesh, get_cameras, get_lights, get_renderer
from Utils.plot import plot_rendered_images, plot_losses, plot_gpu_usage, plot_original_and_generated
from Utils.plot import visualize_3d
from Utils.utils_rwtt import get_obj_files, train_test_split
import pickle

# Analysze time it takes to run the code and memory used
# with clean up inside loop and without it
start_time = time.time()

clean = True

# save path for trained generator
save_path = "Models/generator.pth"
# save train_test split so that we can use this later for test
split_path = "Models/split.pickle"
# debug: set it to false before training, used to visualize the rendered images
debug = False
# batch size used for training
batch_size = 1
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
num_epochs = 1
# Learning rate for optimizers
lr = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# device to train: gpu/cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# create a list of obj files with single texture that we can later iterate over
data_root = "/home/user/data/rwtt"
obj_files = get_obj_files(data_root=data_root)
train_files, test_files = train_test_split(obj_files)
print(f"train_files: {len(train_files)}")
print(f"test_files: {len(test_files)}")

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

criterion = nn.BCELoss()

real_label = 1.
fake_label = 0.

optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

loss_G = []
loss_D = []

print("Training...")
gpu_usage = []

for epoch in range(num_epochs):
    # for each epoch iterate over the obj files and train
    print(f"Epoch: {epoch + 1} / {num_epochs}")
    for obj_filename in obj_files:
        # load obj file: can not load some obj files [formatting issues]: just go to next obj file
        # and proceed
        try:
            mesh = load_objs_as_meshes([obj_filename], device=device)
        except:
            continue
        mesh = normalize_mesh(mesh)

        # faces_uvs = mesh.textures._faces_uvs_list
        faces_uvs = mesh.textures.faces_uvs_list()
        # verts_uvs = mesh.textures._verts_uvs_list
        verts_uvs = mesh.textures.verts_uvs_list()

        # get cameras
        cameras = get_cameras(batch_size=batch_size, device=device)
        # get lights
        lights = get_lights(device=device)
        # get renderer
        renderer = get_renderer(image_size=image_size, cameras=cameras, lights=lights, device=device)

        # render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
        # images = renderer(meshes, cameras=cameras, lights=lights)
        images = renderer(mesh, cameras=cameras, lights=lights)

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
        noise = netG.generate_noise(batch_size=1).to(device)
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
        netG.zero_grad()
        label.fill_(real_label)
        # D got updated, make another pass to create computation graph and then to calculate the gradients
        output = netD(fake_batch).view(-1)
        # calculate loss
        errG = criterion(output, label)
        # calculate gradients
        errG.backward()
        # update generator
        optimizerG.step()

        # some book keeping
        loss_G.append(errG.item())

        # Output training stats
        print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch + 1, num_epochs, errD, errG.item()))

        # clean up
        if clean:
            del mesh, cameras, renderer, verts_uvs, faces_uvs, images, fake_images, texture

        gpu_usage.append(get_gpu_memory_map()[0])

# save the trained generator
torch.save(netG.state_dict(), save_path)

# also pickle and dump the train_files and test_files: so that we can check only test files later:
print("saving split into pickle")
split = {"train_files": train_files, "test_files": test_files}
with open(split_path, "wb") as handle:
    pickle.dump(split, handle)

end_time = time.time()
elapsed_minutes = (end_time - start_time) / 60
print(f"Time: {round(elapsed_minutes, 2)} Minutes")

# plot losses
plot_losses(loss_G, loss_D)

# plot gpu_usage
plot_gpu_usage(gpu_usage=gpu_usage)

# generate one texture image
with torch.no_grad():
    noise = netG.generate_noise(batch_size=1).to(device)
    texture = netG(noise).permute(0, 2, 3, 1)

plt.imshow(texture.squeeze().cpu().numpy())
plt.title("sample texture")
plt.show()

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
mesh.textures = TexturesUV(texture.to(device), faces_uvs=faces_uvs, verts_uvs=verts_uvs)
# render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
image = renderer(mesh, cameras=cameras, lights=lights)

plot_original_and_generated(original, image)

visualize_3d(mesh)
