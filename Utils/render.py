"""
This script will render with original texture and save the images to
respective folder so that we can use them at training time
"""

import torch
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras, PointLights, \
    RasterizationSettings
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles
from pytorch3d.renderer import HardFlatShader
from pytorch3d.datasets import ShapeNetCore
import warnings
import pathlib
import matplotlib.pyplot as plt
import pickle
warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if shapenet: provide the shapenet paths
# shapenet_path = "/Users/kcdharma/data/car_full"
shapenet_path = "/home/kcdharma/data/arete-realsim/car_full"
shapenet_dataset = ShapeNetCore(shapenet_path, synsets=["car"], version=2, load_textures=True,
                                texture_resolution=4)

synset_id = "02958343"
model_ids_path = "Data/model_ids.pickle"
with open(model_ids_path, "rb") as read_file:
    model_ids = pickle.load(read_file)

# rendering parameters
# sample new camera location
distance = 1.0
elevations = [0.0, 30.0, 60.0, 85.0]
azimuths = [0.0, 60.0, 90.0, 250.0, 310.0]
model_id = model_ids[0]

print(f"synset_id: {synset_id}")
print(f"model_id: {model_id}")
print("generating images...")

# create render_images folder
folder_path = shapenet_path + "/" + synset_id + "/" + model_id + "/models/render_imgs"
pathlib.Path(folder_path).mkdir(exist_ok=True)

for azimuth in azimuths:
    for elevation in elevations:
        print(f"generating image: azimuth: {azimuth}, elevation: {elevation}")
        R, T = look_at_view_transform(dist=distance, elev=elevation, azim=azimuth)
        camera_location = camera_position_from_spherical_angles(distance=distance, elevation=elevation,
                                                                azimuth=azimuth,
                                                                degrees=True, device=str(device))
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        raster_settings = RasterizationSettings(image_size=512, blur_radius=0.0, faces_per_pixel=1,
                                                cull_backfaces=True)
        # Place a point light in front of the object: for now let's put at the same location of camera:
        # so the rendered image will get some
        # light, we can also place them at s*camera_location where s is some scalar
        lights = PointLights(location=camera_location, ambient_color=((0.5, 0.5, 0.5),),
                             diffuse_color=((0.4, 0.4, 0.4),),
                             specular_color=((0.1, 0.1, 0.1),), device=str(device))

        rendered_image = shapenet_dataset.render(model_ids=[model_id], device=device, cameras=cameras,
                                                 raster_settings=raster_settings,
                                                 lights=lights, shader_type=HardFlatShader)

        # save in render_imgs folder
        plt.imsave(f"{folder_path}/{azimuth}_{elevation}.png", rendered_image[0, ..., :3].cpu().numpy())
