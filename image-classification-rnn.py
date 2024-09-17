'''
Image Classification with RNN (Recurrent Neural Network)

- with Keras 3 API

'''
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import keras
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))
image_size = x_train.shape[1]
print(image_size)

# convert to one-hot vector, we will use CategoricalCrossentropy loss function.
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# resize and normalize
x_train = np.reshape(x_train,[-1, image_size, image_size])
x_train = x_train.astype('float32') / 255.0
x_test = np.reshape(x_test,[-1, image_size, image_size])
x_test = x_test.astype('float32') / 255.

# hyperparameter
batch_size = 128
drop_rate = 0.20
train_epochs = 25
learning_rate=1e-3

# sequential model
model = keras.Sequential([
    keras.Input(shape=(image_size, image_size)),
    # Fully-connected RNN where the output is to be fed back as the new input.
    # output shape is (batch_size, units)
    keras.layers.SimpleRNN(units=256, dropout=drop_rate),
    keras.layers.Dense(num_labels, activation='softmax')
])

model.summary()

# compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.CategoricalCrossentropy(),
    metrics=[
        keras.metrics.CategoricalAccuracy()
    ],
)

# train the model
model.fit(x_train, y_train, epochs=train_epochs, batch_size=batch_size)


# validate the model on test dataset
_, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("\nRNN Test Accuracy: %.1f%%" % (100.0 * acc))