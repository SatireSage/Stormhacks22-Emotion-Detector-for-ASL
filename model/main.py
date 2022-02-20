import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import (
    Dense,
    Flatten,
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
)
from tensorflow.python.keras.losses import sparse_categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
import numpy as np

# setting variables and directories for training and testing paths
img_size = 224
batch_size = 10000
epochs = 5
train_path = "/archive/asl_alphabet_train/asl_alphabet_train"
test_path = "/archive/asl_alphabet_test/asl_alphabet_test"
# define image data generators for data augmentation and rescaling
augment_train_data = ImageDataGenerator(
    horizontal_flip=True,
    rotation_range=50,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1.0 / 255,
)
augment_test_data = ImageDataGenerator(rescale=1.0 / 255)
# run image data generators on training and testing dataset
train_dataset = augment_train_data.flow_from_directory(
    train_path,
    shuffle=True,
    classes=[
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "space",
        "del",
        "nothing",
    ],
    target_size=(img_size, img_size),
    batch_size=batch_size,
)
test_dataset = augment_train_data.flow_from_directory(
    test_path,
    classes=[
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
        "space",
        "del",
        "nothing",
    ],
    target_size=(img_size, img_size),
    batch_size=batch_size,
)
# getting pretrained model for transfer learning and defining model
url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
download_model = hub.KerasLayer(url, input_shape=(img_size, img_size, 3))
model = Sequential([download_model, Dense(29), Activation("softmax")])
# compiling model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
# training model
print("\n Model Training: ")
model.fit(train_dataset, batch_size=batch_size, epochs=epochs)
print("\n Model summary: ")
print(model.summary())
print("\n Model Evaluation: ")
model.evaluate(test_dataset)
model.save("/archive/asl_model.h5")
load_model = tf.keras.models.load_model(
    "/asl_model.h5",
    custom_objects={"KerasLayer": hub.KerasLayer},
)
print(load_model.summary())
