#this is my code
print("CNN Basics")

import tensorflow as tf   # ML Framework for DL
from tensorflow import keras    #API for building and training deep learning models
from tensorflow.keras import layers # for building neural network layers
import numpy as np # for numerical operations
import matplotlib.pyplot as plt # for data visualization



#Load dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data() # Load CIFAR-10 dataset

# Keep only airplanes (class 0) and cars (class 1)
def filter_classes(x, y, classes):
    mask = np.isin(y[:, 0], classes)
    return x[mask], y[mask]

x_train, y_train = filter_classes(x_train, y_train, [0, 1])
x_test,  y_test  = filter_classes(x_test,  y_test,  [0, 1])


#Normalization, lower the pixel values to the range [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

#model

# Find edges → find shapes → shrink → find complex patterns → shrink → unroll → reason → decide
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),  #[Conv2D]3x3 window (kernel) slides over the image, 32 times(filters) [Relu] Activation funtion, - values set to 0 [inputshape] tells the model to expect 32 x 32px 3 color channels, looks for edges
    layers.MaxPooling2D((2, 2)), #Shrinks the image by half. Looks at every 2×2 block of pixels and keeps only the biggest value. This reduces computation
    layers.Conv2D(64, (3, 3), activation='relu'), #64 filets, looks for more complex features like corners and textures shapes.
    layers.MaxPooling2D((2, 2)), #Shrinks again by half. Same reason as before.
    layers.Flatten(), #flattening the 2d grid so it can be fed into nn 
    layers.Dense(64, activation='relu'), # neural network layer — every input connects to all 64 neurons. This is where the model reasons about the features the conv layers found
    layers.Dense(1, activation='sigmoid')  # Binary classification
])


#set up for training
model.compile(
    optimizer="adam",  #Adam is an optimization algorithm that adjusts the learning rate during training, making it faster and more efficient.
    loss="binary_crossentropy", #Common loss function for binary classification tasks. It measures the difference between the predicted probabilities and the actual labels, encouraging the model to output probabilities close to 0 or 1.
    metrics=["accuracy"] #Accuracy report from keras during training
)

history = model.fit(
    x_train, y_train,
    epochs=10, #go through the entire training dataset 10 times
    batch_size=32, #look at 32 samples at a time before updating the model's weights. This helps with memory efficiency and can lead to faster convergence.
    validation_split=0.2 #set sized reserved for validation.
)
test_loss, test_acc = model.evaluate(x_test, y_test) #evaluate the model on the test set to see how well it generalizes to unseen data
print(f"Test accuracy: {test_acc:.4f}") #print the test accuracy
plt.plot(history.history["accuracy"],     label="train accuracy")
plt.plot(history.history["val_accuracy"], label="val accuracy")  # accuracy on the validation set during training val accurcy > train accuracy is a good sign that the model is generalizing well and not overfitting to the training data
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()