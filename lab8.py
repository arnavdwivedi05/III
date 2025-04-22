# import tensorflow as tf
# from tensorflow.keras.datasets import mnist
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# import matplotlib.pyplot as plt

# # Load and preprocess data
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train[..., None] / 255.0, x_test[..., None] / 255.0
# y_train, y_test = to_categorical(y_train), to_categorical(y_test)

# # Build model
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
#     MaxPooling2D(),
#     Dropout(0.25),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D(),
#     Dropout(0.25),
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dense(10, activation='softmax')
# ])

# # Compile and train
# model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# # Evaluate
# loss, acc = model.evaluate(x_test, y_test)
# print(f"Test Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# # Plot accuracy
# plt.plot(history.history['accuracy'], label='Train')
# plt.plot(history.history['val_accuracy'], label='Test')
# plt.title('Accuracy over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()