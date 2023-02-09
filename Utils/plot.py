"""
All functions related to plotting things like rendered images, losses and more
"""

import matplotlib.pyplot as plt
from Utils.utils import get_vertex_color
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene, AxisArgs


def plot_rendered_images(images):
    """
    plots the rendered images from pytorch3d
    :param images: rendered images
    :return: None
    """

    plt.figure(figsize=(7, 7))
    plt.imshow(images[0, ..., :3].cpu().numpy())
    plt.title("Rendered with texture")
    plt.grid("off")
    plt.axis("off")
    plt.show()


def plot_losses(loss_g, loss_d):
    """
    plots the loss of generator and discriminator
    :param loss_g: generator loss
    :param loss_d: discriminator loss
    :return: None
    """

    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(loss_g, label="G")
    plt.plot(loss_d, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_gpu_usage(gpu_usage):
    """
    plots the gpu usage per epoch
    :param gpu_usage: list of gpu usage
    :return: None
    """
    plt.figure(figsize=(7, 7))
    plt.title("GPU usage")
    plt.plot(gpu_usage)
    plt.xlabel("epoch")
    plt.ylabel("gpu_memory")
    plt.show()


def plot_original_and_generated(original, generated):
    """
    plots two rendered images in same plot for comparision
    :param original: generally these are the images rendered with real texture
    :param generated: generally these are the images rendered with generated texture
    :return: None
    """
    
    plt.figure(figsize=(7, 7))
    plt.subplot(1, 2, 1)
    plt.imshow(original[0, ..., :3].cpu().numpy())
    plt.title("Rendered with original texture")
    plt.grid("off")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(generated[0, ..., :3].cpu().numpy())
    plt.title("Rendered with synthetic texture")
    plt.grid("off")
    plt.axis("off")
    plt.show()


def visualize_3d(mesh, calc_vertex_texture=True):
    """
    render the image for 3d visualization
    pytorch3d doesn't support TextureUV and rotation in 3D [plotly issue]
    Instead sample texture per vertex and apply that one
    :param mesh: mesh object from pytorch3d
    :param calc_vertex_texture: bool: to indicate if we should calculate color per vertex
    :return: None
    """

    if calc_vertex_texture:
        print("Please wait, generating vertex colors from texture...")
        colors = get_vertex_color(mesh)
        verts_rgb = colors.unsqueeze(dim=0)
        textures = TexturesVertex(verts_features=verts_rgb)
        mesh = Meshes(verts=[mesh.verts_packed()], faces=[mesh.faces_packed()], textures=textures)

    fig = plot_scene({"": {"mesh": mesh}},
                     xaxis={"showgrid": False, "zeroline": False, "visible": False},
                     yaxis={"showgrid": False, "zeroline": False, "visible": False},
                     zaxis={"showgrid": False, "zeroline": False, "visible": False},
                     axis_args=AxisArgs(showgrid=False))
    fig.show()
