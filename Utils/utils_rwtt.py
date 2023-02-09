"""
utils for some processing of real world things dataset
"""

import os
import glob
from PIL import Image

# allow to load image of any size
Image.MAX_IMAGE_PIXELS = None


def get_obj_files(data_root):
    """
    get the list of obj files for all folders inside real world textured things
    :param data_root: location of root of textured things dataset
    :return: list of obj files
    """

    chunks = os.listdir(data_root)
    obj_files = []
    total = 0
    for chunk in chunks:
        folders = os.listdir(data_root + "/" + chunk)
        count = 0
        for folder in folders:
            current_path = data_root + "/" + chunk + "/" + folder
            texture_images = glob.glob(current_path + "/" + "*.png")
            if len(texture_images) == 1:
                # check the size of image do not load big textures -- or resize textures to [1024 x 1024]
                im = Image.open(texture_images[0])
                width, height = im.size
                if width <= 4096 and height <= 4096:
                    count += 1
                    # assert we have single obj file
                    obj_file_list = glob.glob(current_path + "/" + "*.obj") + glob.glob(current_path + "/" + "*.OBJ")
                    assert len(obj_file_list) == 1
                    # append the corresponding obj file
                    obj_files.append(obj_file_list[0])

        # print(chunk + " : " + str(count))
        total += count

    print(f"total obj files: {total}")
    assert len(obj_files) == total
    return obj_files


def train_test_split(obj_files):
    """
    split the given list of obj files into train and test obj files
    :param obj_files: list of obj files
    :return: list of train and test obj files
    """

    # create a list of files used for training and files used for testing texture synthesis
    num_test = int(0.045 * len(obj_files))
    train_files = obj_files[0:-num_test]
    test_files = obj_files[-num_test:]
    for file in test_files:
        assert file not in train_files

    return train_files, test_files
