from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array

import os
import glob
import numpy as np
import cv2


def get_list_folders(data_root_folder):
    list_folders = os.listdir(data_root_folder)
    # print(list_files)
    print(str(len(list_folders)) + ' folders found')
    return list_folders


def create_train_data(data_root_dir):
    folder_names = get_image_list(data_root_dir)
    datas = []
    for folder_name in folder_names:
        folder_path = os.path.join(data_dir, folder_name)
        print(folder_path)
        for image_path in glob.glob(os.path.join(folder_path, '*.png')):
            data = cv2.imread(image_path, 0)
            data = data.reshape(data.shape + (1,))
            # print(data.shape)
            data_expanded = np.expand_dims(data, axis=0)
            # print(data_expanded.shape)
            datas.append(data_expanded)
            # print(image_path)
    image_datas = np.concatenate(datas, axis=0)
    # print(image_datas.shape)
    return image_datas


def get_image_list(image_folder):
    list_files = os.listdir(image_folder)
    # print(list_files)
    print(str(len(list_files)) + ' files found')
    return list_files


if __name__ == "__main__":
    data_dir = "../AB/train"
    x_train = create_train_data(data_dir)
    print(x_train.shape)
    print(type(x_train))




