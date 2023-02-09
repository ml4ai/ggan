"""
utilities functions for general use in the project
Ref: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html for weight_init
"""

import torch.nn as nn
import torch
import subprocess
import random
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, RasterizationSettings
from pytorch3d.renderer import MeshRenderer, MeshRasterizer, SoftPhongShader
import torch.nn.functional as f


def weights_init(m):
    """
    weight initialization for parameters of generator and discriminator
    :param m: pytorch module [conv2d, fc] layers
    :return: None
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Vertex:
    """
    class that holds the color[r, g, b] and vertex [x, y, z] positions for each vertex in a given mesh
    """

    def __init__(self, idx, position):
        self.idx = idx
        self.position = position  # [x, y, z]
        self.color = []  # sum of [r, g, b] for all occurrence of vertices
        self.update_count = 0

    def update_color(self, new_color):
        if not self.color:
            self.color = new_color
        else:
            self.color = map(lambda x, y: x + y, self.color, new_color)
        self.update_count += 1


def get_vertex_color(mesh):
    """
    calculate the color of each vertex based on given texture of the mesh
    :param mesh: pytorch3d mesh object
    :return: colors[r, g, b] of each vertex
    """
    vertex_dict = {}
    for idx, pos in enumerate(mesh.verts_packed()):  # V x 3
        vertex_dict[idx] = Vertex(idx, pos.tolist())

    centers = mesh.textures.centers_for_image(index=0).numpy()
    for idx, face in enumerate(mesh.faces_packed()):  # F x 3
        v1_idx, v2_idx, v3_idx = face.tolist()
        # mesh.textures.faces_uvs_list()[0] --> F x 3
        vt1_idx, vt2_idx, vt3_idx = mesh.textures.faces_uvs_list()[0][idx].tolist()

        x1, y1 = centers[vt1_idx]
        color = mesh.textures.maps_padded()[0][int(y1), int(x1)]
        vertex_dict[v1_idx].update_color(color.tolist())

        x2, y2 = centers[vt2_idx]
        color = mesh.textures.maps_padded()[0][int(y2), int(x2)]
        vertex_dict[v2_idx].update_color(color.tolist())

        x3, y3 = centers[vt3_idx]
        color = mesh.textures.maps_padded()[0][int(y3), int(x3)]
        vertex_dict[v3_idx].update_color(color.tolist())

    colors = torch.ones_like(mesh.verts_packed())

    for idx, item in vertex_dict.items():
        temp_colors = list(item.color)
        # some models have vertex color = []
        # need to figure out why
        # temporary fix now
        if len(temp_colors) == 3:
            colors[idx] = torch.FloatTensor([x / item.update_count for x in temp_colors])

    return colors


def get_gpu_memory_map():
    """Get the current gpu usage.
    Ref: https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/3
    Returns: dictionary of gpu usage per device_id
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def normalize_mesh(mesh):
    """
    normalize and center the target mesh so that we can have same light and similar R, T for camera
    that works on every objects
    :param mesh: pytorch3d mesh object
    :return: normalized mesh object
    """

    verts = mesh.verts_packed()
    n = verts.shape[0]
    center = verts.mean(0)
    scale = max((verts - center).abs().max(0)[0])
    mesh.offset_verts_(-center.expand(n, 3))
    mesh.scale_verts_((1.0 / float(scale)))
    return mesh


def get_cameras(batch_size, device):
    """
    sample new camera location
    :param batch_size: batch_size for cameras
    :param device: device_id to copy the camera objects
    :return: batch_size camera objects
    """

    dist = round(random.uniform(2.0, 3.0), 2)
    # Randomization: sample batch_size*2 values for each, randomly permute them and select the batch_size values
    elev_double = torch.linspace(0, 180, batch_size * 2)
    azim_double = torch.linspace(-180, 180, batch_size * 2)

    indices = torch.randperm(batch_size * 2)[:batch_size]
    elev = elev_double[indices]
    azim = azim_double[indices]

    r, t = look_at_view_transform(dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=r, T=t)
    return cameras


def get_lights(device):
    """
    Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    -z direction.
    :param device: device [gpu] to copy the light object
    :return: light object at a given position
    """

    lights = PointLights(device=str(device), location=[[2.0, 0.0, 0.0]])
    return lights


def get_renderer(image_size, cameras, lights, device):
    """
    get renderer with MeshRasterizer and SoftPhongShader and given parameters
    :param image_size: size of image to be rendered
    :param cameras: cameras objects
    :param lights: lights objects
    :param device: location of renderer object
    :return: pytorch3d renderer
    """

    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # Create a phong renderer by composing a rasterizer and a shader. The textured phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device,
            cameras=cameras,
            lights=lights
        )
    )

    return renderer


# helper functions for stylegan
def d_logistic_loss(real_pred, fake_pred):
    real_loss = f.softplus(-real_pred)
    fake_loss = f.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def g_nonsaturating_loss(fake_pred):
    loss = f.softplus(-fake_pred).mean()

    return loss
