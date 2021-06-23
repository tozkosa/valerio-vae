import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory

data_dir = '../AB/train'
batch_size = 32
num_classes = 2


def load_images_from_directory():
    train_ds = image_dataset_from_directory(
        data_dir,
        color_mode='grayscale',
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(28, 28),
        batch_size=batch_size
    )

    val_ds = image_dataset_from_directory(
        data_dir,
        color_mode='grayscale',
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(28, 28),
        batch_size=batch_size
    )

    return train_ds, val_ds


def show_images(train_ds):
    class_names = train_ds.class_names
    print(class_names)

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    for image_batch, labels_batch in train_ds:
        print(image_batch.shape)
        print(labels_batch.shape)
        break


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(num_classes)
    ])

    model.compile(
        optimizer='adam',
        loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['acc']
    )

    return model


def train_model(model, train_data, val_data):
    history = model.fit(
        train_data,
        validation_data=val_data,
        epochs=50
    )
    return history


def show_learning_process(history):
    plt.figure(figsize=(10, 10))
    plt.plot(history.history['acc'], label='acc')
    plt.plot(history.history['val_acc'], label='val_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(loc='best')


if __name__ == "__main__":
    print(tf.__version__)
    train_ds, val_ds = load_images_from_directory()
    show_images(train_ds)
    model = build_model()
    history = train_model(model, train_ds, val_ds)
    show_learning_process(history)
    plt.show()
