'''
Image Classification with CNN (Convolutional Neural Network)

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

# convert to one-hot vector, we will use CategoricalCrossentropy loss function.
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

# resize and normalize
x_train = np.reshape(x_train,[-1, image_height, image_width, 1])
x_train = x_train.astype('float32') / 255.0
x_test = np.reshape(x_test,[-1, image_height, image_width, 1])
x_test = x_test.astype('float32') / 255.0

# hyperparameter
batch_size = 128
drop_rate = 0.20
train_epochs = 5
learning_rate=1e-3

# sequential model
model = keras.Sequential([
    keras.Input(shape=(image_height, image_width, 1)),
    keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Flatten(),
    keras.layers.Dropout(drop_rate),
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
print("\nCNN Test Accuracy: %.1f%%" % (100.0 * acc))