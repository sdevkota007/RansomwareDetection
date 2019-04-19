import tensorflow as tf
import sys
from project_code.model.util import get_seperate
import pickle
from sklearn.utils import shuffle
import numpy as np

# mnist = tf.keras.datasets.mnist  # mnist is a dataset of 28x28 images of handwritten digits and their labels
# (x_train, y_train),(x_test, y_test) = mnist.load_data()  # unpacks images to x_train/x_test and labels to y_train/y_test



with open("train.pickle", "rb") as f:
    reduced = pickle.load(f)
    Xtrain = pickle.load(f)
    Ytrain = pickle.load(f)
# Shuffle data
x_train, y_train = get_seperate(shuffle(zip(reduced, Ytrain)))


with open("test.pickle", "rb") as f:
    reduced = pickle.load(f)
    Xtest = pickle.load(f)
    Ytest = pickle.load(f)

x_test, y_test = get_seperate(shuffle(zip(reduced, Ytest)))


x_train = np.array(x_train, dtype=np.float32)
x_train = x_train.reshape((x_train.shape[0], -1, x_train.shape[1]))
x_test = np.array(x_test, dtype=np.float32)
x_test = x_test.reshape((x_test.shape[0], -1, x_test.shape[1]))

y_train = np.array(y_train, dtype=np.float32)
y_test = np.array(y_test, dtype=np.float32)


print('x_train')
print('Shape', x_train.shape)
print('Type', type(x_train))
print('Type of one training example', type(x_train[0]))


print('x_test')
print('Shape', x_test.shape)
print('Type', type(x_test))
print('Type of one training example', type(x_test[0]))
exit()



x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between 0 and 1

model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # a simple fully-connected layer, 128 units, relu activation
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # our output layer. 10 units for 10 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

model.fit(x_train, y_train, epochs=3)  # train the model

val_loss, val_acc = model.evaluate(x_test, y_test)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy