'''Preprocess the data before loading it into the dataset class'''
import tifffile
import os
from utils.data_format import transform_into_pinhole_2channels, concat_pinholes_ret_azim

n_lenses = 16
n_pix = 16
root_dir = "/mnt/efs/shared_data/restorators/spheres/images"
save_dir = "/mnt/efs/shared_data/restorators/spheres_prc/images"
img_filenames = os.listdir(os.path.join(root_dir))
for filename in img_filenames:
    img_path = os.path.join(root_dir, filename)
    img = tifffile.imread(img_path)
    pin_img = transform_into_pinhole_2channels(img, n_lenses, n_pix)
    concat_pin_img = concat_pinholes_ret_azim(pin_img, n_lenses, n_pix)
    save_path = os.path.join(root_dir, filename)
    tifffile.imwrite(save_path, concat_pin_img)