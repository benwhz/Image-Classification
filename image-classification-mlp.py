'''
Image Classification with MLP (Multilayer Perceptron)


'''

import keras
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# compute the number of labels
num_labels = len(np.unique(y_train))
image_size = x_train.shape[1]
input_size = image_size * image_size

# resize and normalize
x_train = np.reshape(x_train, [-1, input_size])
x_train = x_train.astype('float32') / 255.0
x_test = np.reshape(x_test, [-1, input_size])
x_test = x_test.astype('float32') / 255.
#print(x_train[0].shape)
# we need not convert label to one-hot vector, 
# because we will use the SparseCategoricalCrossentropy, 
# not the CategoricalCrossentropy

# hyperparameter
batch_size = 128
hidden_unit = 256
drop_rate = 0.50
train_epochs = 50

# sequential model
model = keras.Sequential([
    keras.Input(shape=(input_size,)),
    keras.layers.Dense(hidden_unit, activation='relu'),
    keras.layers.Dropout(drop_rate),
    keras.layers.Dense(hidden_unit, activation='relu'),
    keras.layers.Dropout(drop_rate),
    # We specify activation=None so as to return logits
    keras.layers.Dense(num_labels, activation=None)
])

model.summary()

# compile the model
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[
        keras.metrics.SparseCategoricalAccuracy()
    ],
)

# train the network
model.fit(x_train, y_train, epochs=train_epochs, batch_size=batch_size)

# validate the model on test dataset to determine generalization
_, acc = model.evaluate(x_test, y_test,
    batch_size=batch_size,
    verbose=0)
print("\nTest accuracy: %.1f%%" % (100.0 * acc))