# convert TextureUV to TextureVertex and check if get_vertex_color is working properly or not

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import TexturesVertex
from Utils.utils import normalize_mesh, get_lights, get_renderer, get_vertex_color, get_cameras
from Utils.plot import plot_original_and_generated, visualize_3d
from pytorch3d.structures import Meshes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_size = 1024
num_plots = 10

obj_filename = "./Data/cow_mesh/cow.obj"
mesh = load_objs_as_meshes([obj_filename], device=device)

# mesh with textureuv
mesh_tuv = normalize_mesh(mesh)

faces_uvs = mesh_tuv.textures.faces_uvs_list()
verts_uvs = mesh_tuv.textures.verts_uvs_list()

colors = get_vertex_color(mesh)
verts_rgb = colors.unsqueeze(dim=0)
textures = TexturesVertex(verts_features=verts_rgb)

# mesh with vertex texture
mesh_tv = Meshes(verts=[mesh.verts_packed()], faces=[mesh.faces_packed()], textures=textures)

for i in range(num_plots):
    cameras = get_cameras(batch_size=1, device=device)
    # get lights
    lights = get_lights(device=device)
    # get renderer
    renderer = get_renderer(image_size=image_size, cameras=cameras, lights=lights, device=device)

    image_tuv = renderer(mesh_tuv, cameras=cameras, lights=lights)
    # render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
    image_tv = renderer(mesh_tv, cameras=cameras, lights=lights)

    plot_original_and_generated(image_tuv, image_tv)

visualize_3d(mesh_tv, calc_vertex_texture=False)
