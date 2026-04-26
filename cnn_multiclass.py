import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Keep only airplane (0), car (1), ship (8)
def filter_classes(x, y, classes):
    mask = np.isin(y[:, 0], classes)
    return x[mask], y[mask]

x_train, y_train = filter_classes(x_train, y_train, [0, 1, 8])
x_test,  y_test  = filter_classes(x_test,  y_test,  [0, 1, 8])

# Normalize
x_train = x_train.astype("float32") / 255.0
x_test  = x_test.astype("float32")  / 255.0

# One-hot encode labels
y_train = keras.utils.to_categorical(y_train, num_classes=10)
y_test  = keras.utils.to_categorical(y_test,  num_classes=10)

model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax") # Output layer for 10 classes, softmax gives probabilities for each class
])
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy", #multi-class classification loss function, encourages the model to output a probability distribution over the 10 classes
    metrics=["accuracy"]
)

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2
)
model.save(r"C:\Workspace\CNN\backend\models\multiclass_cnn")

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")

plt.plot(history.history["accuracy"],     label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()