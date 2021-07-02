from tensorflow.keras.datasets import mnist
from tensorflow.keras.preprocessing import image_dataset_from_directory
from vae import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 20
data_dir = "../AB/train"


def load_images_from_directory():
    train_ds = image_dataset_from_directory(
        data_dir,
        color_mode='grayscale',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(28, 28),
        batch_size=BATCH_SIZE
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        color_mode='grayscale',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(28, 28),
        batch_size=BATCH_SIZE
    )

    return train_ds, val_ds


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, epochs):
    autoencoder = VAE(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train_loader(train_ds, epochs)

    return autoencoder


if __name__ == "__main__":
    # x_train, _, _, _ = load_mnist()
    train_ds, _ = load_images_from_directory()
    autoencoder = train(train_ds, LEARNING_RATE, EPOCHS)
    autoencoder.save("model")
    autoencoder2 = VAE.load("model")
    autoencoder2.summary()
