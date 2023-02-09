"""
train our model using graph neural network that predicts colors per vertex as a texture map
[later: texture image per face]
"""
import torch
from pytorch3d.structures import Meshes
from Utils.utils import weights_init
from torch import nn
from torch import optim
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, TexturesVertex, TexturesAtlas
from pytorch3d.renderer import HardFlatShader
from pytorch3d.datasets import ShapeNetCore
import warnings
from Utils.plot import plot_original_and_generated
import argparse
from Models.resgcnface import ResGCNFace
from Models.nerf import NeRF
from Models.gcn import GCN
import json
from Utils.mesh_utils import mesh_features, mesh_features_dual
from tqdm import tqdm
from Utils.fid import get_fid
import matplotlib.pyplot as plt
import lpips
from Utils.utils import normalize_mesh
from PIL import UnidentifiedImageError
import time

warnings.filterwarnings("ignore")

# parse the arguments for config of Generator and config of Discriminator
parser = argparse.ArgumentParser()
parser.add_argument("--config", help="configuration file for paths and hyperparameters")
# parser.add_argument("--configG", help="configuration file for Generator")
parser.add_argument("--configD", help="configuration file for Discriminator")
args = parser.parse_args()

config = json.load(open(args.config, 'r'))
# parse config file and get the parameters
# learning rate for gnn
lr = config["lr"]
# weight decay for adam optimizer [gnn]
weight_decay = config["weight_decay"]

# output texture resolution for textureatlas
out_res = config["out_res"]

dual = config["dual"]
if dual:
    # 6 for now [x, y, z] for the face and [n_x, n_y, n_z] for the face
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

dropout = config["dropout"]
# gpu_index to train on
gpu_index = config["gpu_index"]
# batch size used for training
batch_size = config["batch_size"]
# device to train: gpu/cpu
device = torch.device("cuda:" + str(gpu_index) if torch.cuda.is_available() else "cpu")
save_folder = config["save_folder"]
# save path for trained generator
model_type = config["model_type"]
print(f"model_type: {model_type}")

save_path_generator = save_folder + "/" + model_type + ".pt"
# in case of shapenet - specify category
category = config["category"]
# noise dimension
noise_dim = config["noise_dim"]

# if debug: will show the image rendered with original texture and rendered with synthetic texture during
# training: disable on kraken, can be enabled on local machine for debugging
debug = config["debug"]

# spatial size of image generated from generator/ input to Discriminator / rendered
image_size = config["image_size"]

if image_size == 64:
    from Discriminator.discriminator64 import Discriminator
elif image_size == 128:
    from Discriminator.discriminator128 import Discriminator
elif image_size == 512:
    from Discriminator.discriminator512 import Discriminator
else:
    from Discriminator.discriminator1024 import Discriminator

print("Initializing Discriminator...")
netD = Discriminator(args.configD).to(device)
print(netD)

print("Initializing weights for Generator and Discriminator...")
# weight initialization for GCN is done inside the model itself
# netG.apply(weights_init)
netD.apply(weights_init)

# Beta hyperparam for Adam optimizers
beta1 = config["beta1"]
beta2 = config["beta2"]

wgan = config["wgan"]
use_image_loss = config["image_loss"]
gradient_penalty = config["gradient_penalty"]
lambda_gp = config["lambda_gradient_penalty"]
p_loss_weight = config["p_loss_weight"]

criterion = nn.BCELoss()
real_label = 1.
fake_label = 0.

# try wgan loss
clamp_lower = config["clamp_lower"]
clamp_upper = config["clamp_upper"]
lr_wgan = config["lr_wgan"]

# optimizer for Discriminator
if wgan:
    optimizerD = optim.RMSprop(netD.parameters(), lr=lr_wgan)
else:
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))

# num epochs to train
num_epochs = config["num_epochs"]

# if shapenet: provide the shapenet paths
shapenet_path = config["shapenet_path"]
texture_loss = config["texture_loss"]
texture_loss_weight = config["texture_loss_weight"]

if texture_loss:
    shapenet_dataset = ShapeNetCore(shapenet_path, synsets=[category], version=2, load_textures=True,
                                    texture_resolution=out_res)
else:
    shapenet_dataset = ShapeNetCore(shapenet_path, synsets=[category], version=2, load_textures=True,
                                    texture_resolution=4)

# loss for generator [plot]
loss_G = []
# loss for discriminator [plot]
loss_D = []
# val fid [plot]
val_fids = []

print("Training...")
best_fid = 1000
p_loss = config["p_loss"]
if p_loss:
    loss_fn_alex = lpips.LPIPS(net='alex')

# netG and netD
print("Initializing Generator...")
if model_type == "resgcnface":
    netG = ResGCNFace(in_features=in_features, n_hidden=n_hidden, out_features=out_features,
                      noise_dim=noise_dim, dropout=dropout).to(device)
elif model_type == "nerf":
    netG = NeRF(in_features=in_features, n_hidden=n_hidden, out_features=out_features,
                noise_dim=noise_dim, dropout=dropout).to(device)
else:
    netG = GCN(in_features=in_features, n_hidden=n_hidden, out_features=out_features,
               noise_dim=noise_dim, dropout=dropout).to(device)

netG.apply(weights_init)

netG.train()
print(netG)

best_model = netG

# optimizer for Generator
optimizerG = optim.Adam(netG.parameters(), lr=lr, weight_decay=weight_decay)
# color_tensor = torch.tensor(temp).to(device)

total_samples = len(shapenet_dataset)
train_set = list(range(total_samples - 200))
val_set = list(range(total_samples - 200, total_samples - 100))
test_set = list(range(total_samples - 100, total_samples))

print(f"training samples: {len(train_set)}")
print(f"val samples: {len(val_set)}")
print(f"test samples: {len(test_set)}")

print("Evaluating val fid from random initial model...")
val_fid = get_fid(shapenet_path, netG, val_set, device, dual, out_res, category=category,
                  image_size=image_size, noise_dim=noise_dim)
print(f"val_fid from random model: {round(val_fid, 4)}")

print("Training...")
for epoch in range(num_epochs):
    start_time = time.time()
    # for each epoch iterate over the obj files and train
    print(f"Epoch: {epoch + 1} / {num_epochs}")
    errorD_total = 0.0
    errorG_total = 0.0

    # set both models to train mode
    netG.train()
    netD.train()

    counter = 0
    for idx in tqdm(train_set):
        try:
            shapenet_model = shapenet_dataset[idx]
            synset_id = shapenet_model["synset_id"]
            model_id = shapenet_model["model_id"]
            original_textures = shapenet_model["textures"].unsqueeze(dim=0).to(device)
            verts = shapenet_model["verts"]
            faces = shapenet_model["faces"]
            mesh = Meshes([verts.to(device)], [faces.to(device)])
            mesh = normalize_mesh(mesh)
            save_path = shapenet_path + "/" + synset_id + "/" + model_id + "/models"
            if dual:
                features, adj = mesh_features_dual(save_path=save_path, mesh=mesh)
            else:
                features, adj = mesh_features(save_path=save_path)
            features = features.to(device)
            adj = adj.to(device)
            counter += 1
        except (IndexError, UnidentifiedImageError):
            continue

        # sample new camera location
        distance = torch.max(mesh.verts_packed().abs()).item() + 1
        elevation = torch.FloatTensor(batch_size).uniform_(0, 180)
        azimuth = torch.FloatTensor(batch_size).uniform_(-180, 180)
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
                                                cull_backfaces=True)
        lights = PointLights(location=[[1.0, 1.0, 1.0]], ambient_color=((0.5, 0.5, 0.5),),
                             diffuse_color=((0.4, 0.4, 0.4),),
                             specular_color=((0.1, 0.1, 0.1),), device=str(device))
        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                                shader=HardFlatShader(cameras=cameras, lights=lights, device=device)).to(device)
        if dual:
            mesh.textures = TexturesAtlas(original_textures)
        else:
            mesh.textures = TexturesVertex(verts_features=original_textures)

        meshes = mesh.extend(batch_size).to(device)
        rendered_images = renderer(meshes)

        # zero grad the netD
        netD.zero_grad()
        # extract RGB images from these real rendered images
        real_batch = rendered_images[..., :3].to(device).permute(0, 3, 1, 2)
        if not wgan:
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            # forward pass the real batch through D
            output = netD(real_batch).view(-1)
            errD_real = criterion(output, label)
            # calculate gradients for real batch
            errD_real.backward()

        # generate noise
        noise = torch.randn(noise_dim).to(device)
        texture = netG(features, adj, noise).to(device).unsqueeze(dim=0)
        if dual:
            texture = torch.reshape(texture, (texture.shape[0],
                                              texture.shape[1],
                                              out_res, out_res,
                                              3)).to(device)
            mesh.textures = TexturesAtlas(texture)
        else:
            mesh.textures = TexturesVertex(verts_features=texture)

        # will produce fake images of shape [batch_sizeximage_sizeximage_sizex4(RGBA)]
        meshes = mesh.extend(batch_size).to(device)
        fake_images = renderer(meshes)
        fake_batch = fake_images[..., :3].to(device).permute(0, 3, 1, 2)

        if debug:
            plot_original_and_generated(rendered_images, fake_images.detach())

        if not wgan:
            label.fill_(fake_label)
            # pass fake batch to netD
            output = netD(fake_batch.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
        else:
            d_loss = torch.mean(netD(fake_batch.detach())) - torch.mean(netD(real_batch))
            d_loss.backward()

        if gradient_penalty:
            eps = torch.randn([batch_size, 1, 1, 1], device=device)
            x_hat = eps * real_batch.detach() + (1 - eps) * fake_batch.detach()
            x_hat = x_hat.to(device)
            x_hat.requires_grad = True
            out_hat = netD(x_hat)
            gradients = torch.autograd.grad(outputs=out_hat, inputs=x_hat,
                                            grad_outputs=torch.ones_like(out_hat),
                                            create_graph=True,
                                            retain_graph=True,
                                            only_inputs=True)[0]

            gradients = gradients.reshape(gradients.shape[0], -1)
            grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
            grad_loss = lambda_gp * grad_penalty
            grad_loss.backward()

        # update the discriminator
        optimizerD.step()

        if wgan:
            for p in netD.parameters():
                p.data.clamp_(clamp_lower, clamp_upper)

        # list some values for plot
        if wgan:
            errD = d_loss.item()
        else:
            errD = errD_real.item() + errD_fake.item()

        errorD_total += errD

        # Train generator
        # zero grad the generator
        netG.zero_grad()

        if not wgan:
            label.fill_(real_label)
            # D got updated, make another pass to create computation graph and then to calculate the gradients
            output = netD(fake_batch).view(-1)
            # calculate loss: gan loss
            errG = criterion(output, label)
        else:
            errG = -torch.mean(netD(fake_batch))

        # calculate gradients
        if p_loss:
            perceptual_loss = loss_fn_alex(real_batch.cpu(), fake_batch.cpu(), normalize=True).squeeze()
            errG += p_loss_weight * torch.mean(perceptual_loss)

        if use_image_loss:
            errG += torch.mean(torch.abs(real_batch - fake_batch))

        if texture_loss:
            generated = texture[0].reshape(-1, 3).to(device)
            target = original_textures[model_id].reshape(-1, 3).to(device)
            loss_texture = torch.sum((generated - target) ** 2)
            errG += texture_loss_weight * loss_texture

        errG.backward()

        # update generator
        optimizerG.step()

        # some book keepingß
        errorG_total += errG.item()

    # Output training stats
    errorD_total = errorD_total / counter
    errorG_total = errorG_total / counter

    loss_D.append(errorD_total)
    loss_G.append(errorG_total)
    print('[%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f' % (epoch + 1, num_epochs, errorD_total, errorG_total))

    # calculate FID on val set
    val_fid = get_fid(shapenet_path, netG, val_set, device, dual, out_res, category=category,
                      image_size=image_size, noise_dim=noise_dim)
    print(f"val_fid: {round(val_fid, 4)}")

    if val_fid < best_fid:
        best_fid = val_fid
        # best_model = netG
        torch.save(netG.state_dict(), save_path_generator)

    val_fids.append(val_fid)

    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60.0
    print(f"Time [minutes]: {round(elapsed_minutes, 4)}")

# save the best trained generator
best_model.load_state_dict(torch.load(save_path_generator))

# calculate FID on test set
test_fid = get_fid(shapenet_path, netG, test_set, device, dual, out_res, category=category,
                   image_size=image_size, noise_dim=noise_dim)

print(f"test_fid: {round(test_fid, 4)}")

# save losses fig
fig = plt.figure(figsize=(10, 5))
plt.title("Loss During Training")
plt.plot(loss_G, label="G")
plt.plot(loss_D, label="D")

plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig(save_folder + "/" + "loss.png")
plt.close(fig)

# save val_fid fig
fig = plt.figure(figsize=(10, 5))
plt.title("val_fid")
plt.plot(val_fids, label="val_fid")
plt.xlabel("iterations")
plt.ylabel("val_fid")
plt.legend()
plt.savefig(save_folder + "/" + "val_fid.png")
plt.close(fig)
