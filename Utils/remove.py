"""
Remove old features and adj files in case of code changes
"""
import os

shapenet_path = "/home/kcdharma/data/arete-realsim/car_full"
# shapenet_path = "/Users/kcdharma/data/car_full"
synset_id = "02958343"  # car

model_ids = os.listdir(shapenet_path + "/" + synset_id)

for model_id in model_ids:
    folder_path = shapenet_path + "/" + synset_id + "/" + model_id + "/models"
    feature_path = folder_path + "/" + "features_dual.npy"
    adj_path = folder_path + "/" + "adj_dual.npz"
    if os.path.exists(feature_path):
        print(f"removing: {feature_path}")
        os.remove(feature_path)
    if os.path.exists(adj_path):
        print(f"removing: {adj_path}")
        os.remove(adj_path)
