'''
Image Classification with MLP (Multilayer Perceptron)

- with Keras 3 API

'''
import os
os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
import numpy as np

# load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))
image_height = image_width = x_train.shape[1]
input_size = image_height * image_width

# we need not convert label to one-hot vector, 
# because we will use SparseCategoricalCrossentropy loss.

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255.0
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255.0

# hyperparameter
batch_size = 128
drop_rate = 0.40
train_epochs = 20
learning_rate=1e-3

# sequential model
model = keras.Sequential([
    keras.Input(shape=(input_size,)),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(drop_rate),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(drop_rate),
    # We specify activation=None so as to return logits
    keras.layers.Dense(num_labels, activation=None)
])

model.summary()

# compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy()
    ],
)

# train the model
model.fit(x_train, y_train, epochs=train_epochs, batch_size=batch_size)

# validate the model on test dataset
_, acc = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0)
print("\nMLP Test Accuracy: %.1f%%" % (100.0 * acc))