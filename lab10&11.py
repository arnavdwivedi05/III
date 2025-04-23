###### Implement and apply optimization methods for neural networks (AdaGrad, RMSProp, Adam) on any relevant dataset. #######

# import tensorflow as tf
# from tensorflow.keras import layers, models, optimizers
# from tensorflow.keras.datasets import mnist

# # Load and preprocess MNIST
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0

# # Model definition
# def get_model():
#     return models.Sequential([
#         layers.Flatten(input_shape=(28, 28)),
#         layers.Dense(128, activation='relu'),
#         layers.Dense(64, activation='relu'),
#         layers.Dense(10, activation='softmax')
#     ])

# # Training and evaluation
# def train_model(optimizer):
#     model = get_model()
#     model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     for epoch in range(5):
#         model.fit(x_train, y_train, epochs=1, verbose=0)
#         print(f"Epoch {epoch+1} completed")

#     _, acc = model.evaluate(x_test, y_test, verbose=0)
#     print(f"Accuracy with {optimizer.__class__.__name__}: {acc * 100:.2f}%\n")

# # Optimizers to test
# optimizers_list = [optimizers.Adagrad(), optimizers.RMSprop(), optimizers.Adam()]

# for opt in optimizers_list:
#     train_model(opt)



######## Apply, train and visualize Different deep CNN architectures like LeNet, AlexNet on MNIST datasets. ##########


# import tensorflow as tf
# from tensorflow.keras import layers, models
# import matplotlib.pyplot as plt

# # Load and normalize MNIST
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
# x_train = x_train[..., None] / 255.0
# x_test = x_test[..., None] / 255.0

# # LeNet model
# def get_lenet():
#     return models.Sequential([
#         layers.Conv2D(6, 5, activation='relu', input_shape=(28, 28, 1)),
#         layers.MaxPooling2D(),
#         layers.Conv2D(16, 5, activation='relu'),
#         layers.MaxPooling2D(),
#         layers.Flatten(),
#         layers.Dense(120, activation='relu'),
#         layers.Dense(84, activation='relu'),
#         layers.Dense(10, activation='softmax')
#     ])

# # AlexNet-like (small version for MNIST)
# def get_alexnet():
#     return models.Sequential([
#         tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(28, 28, 1)),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Conv2D(128, 3, activation='relu'),
#         tf.keras.layers.MaxPooling2D(),
#         tf.keras.layers.Conv2D(256, 3, activation='relu'),
#         tf.keras.layers.Flatten(),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.Dense(10, activation='softmax')
#     ])


# # Train + Evaluate
# def train_model(model_fn, name):
#     model = model_fn()
#     model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#     for epoch in range(5):
#         model.fit(x_train, y_train, epochs=1, batch_size=64, verbose=0)
#         print(f"Epoch {epoch+1} completed")

#     loss, acc = model.evaluate(x_test, y_test, verbose=0)
#     print(f"Accuracy with {name}: {acc * 100:.2f}%\n")
#     return model

# # Visualize filters
# def show_filters(model):
#     for layer in model.layers:
#         if isinstance(layer, layers.Conv2D):
#             filters = layer.get_weights()[0]
#             fig, axs = plt.subplots(1, 6, figsize=(12, 2))
#             for i in range(6):
#                 axs[i].imshow(filters[:, :, 0, i], cmap='gray')
#                 axs[i].axis('off')
#             plt.show()
#             break

# # Run for both models
# for model_fn, name in [(get_lenet, "LeNet"), (get_alexnet, "AlexNet")]:
#     trained_model = train_model(model_fn, name)
#     show_filters(trained_model)


# Code for placesnet and VGG16

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score

# Load MNIST and preprocess
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.image.resize(tf.stack([x_train]*3, axis=-1), [32, 32]) / 255.0
x_test = tf.image.resize(tf.stack([x_test]*3, axis=-1), [32, 32]) / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# VGG16 Model
def get_vgg16():
    base = VGG16(include_top=False, weights='imagenet', input_shape=(32, 32, 3))
    base.trainable = False
    model = models.Sequential([
        base,
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# PlacesNet-like Model
def get_placesnet():
    model = models.Sequential([
        layers.Conv2D(64, 3, activation='relu', padding='same', input_shape=(32,32,3)),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    return model

# Train and evaluate
def train_model(model_fn, name):
    model = model_fn()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train_cat, epochs=3, batch_size=64, verbose=0)
    y_pred = model.predict(x_test).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy with {name}: {acc*100:.2f}%")

# Run both
for fn, name in [(get_vgg16, "VGG16"), (get_placesnet, "PlacesNet")]:
    train_model(fn, name)
