"""
calculates the fid score between multiple views of a model with original texture and multiple views of a model
with synthetic texture
generated from our generator
Ref: https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/
Ref: https://github.com/mseitzer/pytorch-fid
Ref: https://arxiv.org/pdf/1706.08500.pdf : GANs Trained by a Two Time-Scale Update Rule
Converge to a Local Nash Equilibrium
calculates the FID for given validation and test dataset
"""

import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer import TexturesVertex, RasterizationSettings, TexturesAtlas
from pytorch3d.structures import Meshes
from pytorch3d.datasets import ShapeNetCore
from pytorch3d.renderer.lighting import PointLights
from pytorch3d.renderer import MeshRasterizer, MeshRenderer, HardFlatShader
import warnings
from Utils.inception import InceptionV3
import numpy as np
from scipy.linalg import sqrtm
from Utils.mesh_utils import mesh_features, mesh_features_dual
from Utils.utils import normalize_mesh
from PIL import UnidentifiedImageError

warnings.filterwarnings("ignore")

# initialize device for FID calculation
inception_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# initialize the model
dims = 2048  # extract 2048 dimensional features from inceptionv3 network
block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
inception_model = InceptionV3([block_idx]).to(device=inception_device)
inception_model.eval()


# fid related methods
def get_activations(img_tensor):
    """
    input image tensors and get activations from inceptionv3 network
    :param img_tensor: tensors of shape [batch_size * channels * height * width] [values in 0.0 - 1.0]
    :return: feature vectors from inceptionv3 network [batch_size * dims]
    """

    with torch.no_grad():
        pred = inception_model(img_tensor)[0]
    pred = pred.squeeze(3).squeeze(2).cpu().numpy()
    return pred


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)


def calculate_activation_statistics(img_tensor):
    """Calculation of the statistics used by the FID.
    Params:
    img_tensor: tensors of shape [batch_size * channels * height * width] [values in 0.0 - 1.0]
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    """
    act = get_activations(img_tensor=img_tensor)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def get_fid(shapenet_path, model, model_ids, device, dual, out_res, category, image_size=512, noise_dim=8):
    """
    shapenet_path: path of shapenet
    device: device used for fid calculation cpu/gpu
    model: trained model used for synthetic texture generation
    model_ids: list of model_ids from shapenet to calculate the fid from above
    trained model
    out_res: output resolution of generated texture
    image_size: size of image rendered
    return: fid of the given dataset
    we use mean as an aggregation function
    """
    # set model to eval mode
    model.eval()
    shapenet_dataset = ShapeNetCore(shapenet_path, synsets=[category], version=2, load_textures=True,
                                    texture_resolution=4)

    real_images = []
    generated_images = []

    for model_id in model_ids:
        try:
            shapenet_model = shapenet_dataset[model_id]
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
        except (IndexError, UnidentifiedImageError):
            continue

        distance = torch.max(mesh.verts_packed().abs()).item() + 1
        elevation = torch.tensor([15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]).to(device)
        azimuth = torch.tensor([0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]).to(device)
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        lights = PointLights(location=[[1.0, 1.0, 1.0]], ambient_color=((0.5, 0.5, 0.5),),
                             diffuse_color=((0.4, 0.4, 0.4),),
                             specular_color=((0.1, 0.1, 0.1),), device=str(device))
        raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0,
                                                faces_per_pixel=1,
                                                cull_backfaces=True)
        renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
                                shader=HardFlatShader(cameras=cameras, lights=lights, device=device))

        # generate one texture image
        with torch.no_grad():
            noise = torch.randn(noise_dim).to(device)
            texture = model(features, adj, noise).to(device).unsqueeze(dim=0)

        if dual:
            mesh.textures = TexturesAtlas(original_textures)
        else:
            mesh.textures = TexturesVertex(verts_features=original_textures)

        # render with original texture
        batch_size = elevation.shape[0]
        meshes = mesh.extend(batch_size).to(device)
        real = renderer(meshes)

        real_images.append(real[..., :3].permute(0, 3, 1, 2))

        if dual:
            texture = torch.reshape(texture, (texture.shape[0],
                                              texture.shape[1],
                                              out_res, out_res,
                                              3)).to(device)
            mesh.textures = TexturesAtlas(texture)
        else:
            mesh.textures = TexturesVertex(verts_features=texture)

        meshes = mesh.extend(batch_size).to(device)
        # render images: will produce a tensor of shape: [batch_size, image_size, image_size, 4(RGBA)]
        generated = renderer(meshes)
        generated_images.append(generated[..., :3].permute(0, 3, 1, 2))

    real = torch.cat(real_images, dim=0).to(device)
    generated = torch.cat(generated_images, dim=0).to(device)

    m1, s1 = calculate_activation_statistics(real)
    m2, s2 = calculate_activation_statistics(generated)
    agg = calculate_frechet_distance(mu1=m1, mu2=m2, sigma1=s1, sigma2=s2)
    mean_fid = agg / len(model_ids)
    return mean_fid
