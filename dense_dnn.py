import tensorflow as tf
import sys
from util import get_seperate
import pickle
from sklearn.utils import shuffle
import numpy as np
from collections import Counter
import tensorboard



with open("train.pickle", "rb") as f:
    reduced = pickle.load(f)
    Xtrain = pickle.load(f)
    Ytrain = pickle.load(f)
# Shuffle data
x_train, y_train = get_seperate(shuffle(zip(reduced, Ytrain)))


#convert training set to numpy array
x_train = np.array(x_train, dtype=np.float32)
x_train = x_train.reshape((x_train.shape[0], -1, x_train.shape[1]))
y_train = np.array(y_train, dtype=np.float32)


#normalize, scales data between 0 and 1
x_train = tf.keras.utils.normalize(x_train, axis=1)

num_training_examples = Counter(y_train).values()
print("No. of training examples")
print("Normal: ", num_training_examples[0])
print("Malware: ", num_training_examples[1])



#prepare validation set from training set
validation_size = int(x_train.shape[0] * 0.15)  # percent of training features to use as validation set; 0 to disable validation set
x_validation = x_train[:validation_size]
y_validation = y_train[:validation_size]
x_train = x_train[validation_size:]
y_train = y_train[validation_size:]


model = tf.keras.models.Sequential()  # a basic feed-forward model
model.add(tf.keras.layers.Flatten())  # takes our 28x28 and makes it 1x784
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))  # a simple fully-connected layer
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(32, activation=tf.nn.relu))  # a simple fully-connected layer
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2, activation=tf.nn.softmax))  # our output layer. 2 units for 2 classes. Softmax for probability distribution

model.compile(optimizer='adam',  # Good default optimizer to start with
              loss='sparse_categorical_crossentropy',  # how will we calculate our "error." Neural network aims to minimize loss.
              metrics=['accuracy'])  # what to track

model.fit(x_train, y_train, batch_size=32, epochs=3)  # train the model

val_loss, val_acc = model.evaluate(x_validation, y_validation)  # evaluate the out of sample data with model
print(val_loss)  # model's loss (error)
print(val_acc)  # model's accuracy