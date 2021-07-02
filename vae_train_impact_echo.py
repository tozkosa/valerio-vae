# from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
from vae import VAE
import matplotlib.pyplot as plt
import analysis_vae

import os
import glob
import numpy as np
import cv2

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100


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
    image_datas = image_datas.astype("float32") / 255.
    # print(image_datas.shape)
    return image_datas


def get_image_list(image_folder):
    list_files = os.listdir(image_folder)
    # print(list_files)
    print(str(len(list_files)) + ' files found')
    return list_files


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)

    return autoencoder


def plot_reconstructed_images_w_label(images, reconstructed_images, label):
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


if __name__ == "__main__":
    data_dir = "../AB/train"
    x_train = create_train_data(data_dir)
    print(x_train.shape)
    print(type(x_train))

    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    # autoencoder.save("model")
    # autoencoder2 = Autoencoder.load("model")
    # autoencoder2.summary()

    a = np.zeros(200)
    b = np.ones(200)
    y_train = np.hstack((a, b))

    # print(y_train)

    num_sample_images_to_show = 8
    sample_images, _ = analysis_vae.select_images(x_train, y_train, num_sample_images_to_show)
    reconstructed_images, _ = autoencoder.reconstruct(sample_images)
    plot_reconstructed_images_w_label(sample_images, reconstructed_images, y_train)

    num_images = 400
    sample_images, sample_labels = analysis_vae.select_images(x_train, y_train, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    # analysis.plot_reconstructed_images(sample_images, reconstructed_images)
    analysis_vae.plot_images_encoded_in_latent_space(latent_representations, sample_labels)




