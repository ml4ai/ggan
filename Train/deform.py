"""
I have generated an icosphere with level 5 subdivisions from blender and we have a fixed uv mapping for this sphere
this code will deform the vertices of this icosphere such that the shape of the final model will be similar to provided
target mesh and the final mesh will be an approximation of the provided target mesh
the idea is to use this approximated meshes for uv mapping instead of original mesh so that the mapping function remains constant
Ref: https://pytorch3d.org/tutorials/deform_source_mesh_to_target_mesh
"""

import os
import torch
from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency,
)
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['figure.dpi'] = 100

# Set the device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
    print("WARNING: CPU only, this will be slow!")

# trg_obj = os.path.join('/home/user/data/shapenet_car/02958343/86c8a3137a716d70e742b0b5e87bec54/models/bpy_model.obj')
trg_obj = os.path.join('/home/user/data/car_full/02958343/e738466fb6cc90530714334794526d4/models/bpy_model.obj')

verts, faces, aux = load_obj(trg_obj)
faces_idx = faces.verts_idx.to(device)
verts = verts.to(device)

# We scale normalize and center the target mesh to fit in a sphere of radius 1 centered at (0,0,0).
# (scale, center) will be used to bring the predicted mesh to its original center and scale
# Note that normalizing the target mesh, speeds up the optimization but is not necessary!
center = verts.mean(0)
verts = verts - center
scale = max(verts.abs().max(0)[0])
verts = verts / scale

# We construct a Meshes structure for the target mesh
trg_mesh = Meshes(verts=[verts], faces=[faces_idx])

# We initialize the source shape to be a sphere of radius 1
# src_mesh = ico_sphere(5, device)
src_obj = "/home/user/data/icosphere.obj"
verts, faces, _ = load_obj(src_obj, device=device)
src_mesh = Meshes(verts=[verts.to(device)], faces=[faces.verts_idx.to(device)])


# ###  Visualize the source and target meshes

def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 50000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)
    figure = plt.figure(figsize=(5, 5))
    axis = Axes3D(figure)
    axis.scatter3D(x, z, -y)
    axis.set_xlabel('x')
    axis.set_ylabel('z')
    axis.set_zlabel('y')
    axis.set_title(title)
    axis.view_init(190, 30)
    plt.show()


plot_pointcloud(trg_mesh, "Target mesh")
plot_pointcloud(src_mesh, "Source mesh")

# ## 3. Optimization loop
# We will learn to deform the source mesh by offsetting its vertices
# The shape of the deform parameters is equal to the total number of vertices in src_mesh
deform_verts = torch.full(src_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)

# The optimizer
optimizer = torch.optim.SGD([deform_verts], lr=0.5, momentum=0.9)

# Number of optimization steps
Niter = 50000
# Weight for the chamfer loss
w_chamfer = 1.0  # default: 1.0
# Weight for mesh edge loss
w_edge = 0.01  # default: 1.0
# Weight for mesh normal consistency
w_normal = 0.01  # default: 0.01
# Weight for mesh laplacian smoothing
w_laplacian = 0.01  # default: 0.1

loop = tqdm(range(Niter))

chamfer_losses = []
laplacian_losses = []
edge_losses = []
normal_losses = []

for i in loop:
    # Initialize optimizer
    optimizer.zero_grad()

    # Deform the mesh
    new_src_mesh = src_mesh.offset_verts(deform_verts)

    # We sample 5k points from the surface of each mesh
    sample_trg = sample_points_from_meshes(trg_mesh, 50000)
    sample_src = sample_points_from_meshes(new_src_mesh, 50000)

    # We compare the two sets of pointclouds by computing (a) the chamfer loss
    loss_chamfer, _ = chamfer_distance(sample_trg, sample_src)

    # and (b) the edge length of the predicted mesh
    loss_edge = mesh_edge_loss(new_src_mesh)

    # mesh normal consistency
    loss_normal = mesh_normal_consistency(new_src_mesh)

    # mesh laplacian smoothing
    loss_laplacian = mesh_laplacian_smoothing(new_src_mesh, method="uniform")

    # Weighted sum of the losses
    loss = loss_chamfer * w_chamfer + loss_edge * w_edge + loss_normal * w_normal + loss_laplacian * w_laplacian

    # Print the losses
    loop.set_description('total_loss = %.6f' % loss)

    # Save the losses for plotting
    chamfer_losses.append(float(loss_chamfer.detach().cpu()))
    edge_losses.append(float(loss_edge.detach().cpu()))
    normal_losses.append(float(loss_normal.detach().cpu()))
    laplacian_losses.append(float(loss_laplacian.detach().cpu()))

    # Optimization step
    loss.backward()
    optimizer.step()

# ## 4. Visualize the loss
fig = plt.figure(figsize=(13, 5))
ax = fig.gca()
ax.plot(chamfer_losses, label="chamfer loss")
ax.plot(edge_losses, label="edge loss")
ax.plot(normal_losses, label="normal loss")
ax.plot(laplacian_losses, label="laplacian loss")
ax.legend(fontsize="16")
ax.set_xlabel("Iteration", fontsize="16")
ax.set_ylabel("Loss", fontsize="16")
ax.set_title("Loss vs iterations", fontsize="16")
plt.show()

# Deform the mesh
new_src_mesh = src_mesh.offset_verts(deform_verts)

# ## 5. Save the predicted mesh
# Fetch the verts and faces of the final predicted mesh
final_verts, final_faces = new_src_mesh.get_mesh_verts_faces(0)

# Scale normalize back to the original target size
final_verts = final_verts * scale + center

# Store the predicted mesh using save_obj
final_obj = os.path.join('/home/user/data/', 'final_model_21.obj')
save_obj(final_obj, final_verts, final_faces)
