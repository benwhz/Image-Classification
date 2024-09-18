'''
Image Classification with Transformer-Encoder

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
timesteps = x_train.shape[1]
feature = x_train.shape[2]

# we need not convert label to one-hot vector, 
# because we will use SparseCategoricalCrossentropy loss.

# normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# hyperparameter
batch_size = 128
drop_rate = 0.2
train_epochs = 25
learning_rate=0.005
filters = 64

# transformer encoder layer
class TransformerEncoder(keras.layers.Layer):
    def __init__(self, num_heads, filters, feature, dropout=0.0):
        super().__init__()
        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=feature, dropout=dropout)
        #self.attention = keras.layers.AdditiveAttention()
        self.dropout = keras.layers.Dropout(dropout)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv1d1 = keras.layers.Conv1D(filters=filters, kernel_size=1, activation="relu")
        self.conv1d2 = keras.layers.Conv1D(filters=feature, kernel_size=1)
        #self.liner1 = keras.layers.Dense(256)
        #self.liner2 = keras.layers.Dense(feature)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        # Attention and Normalization
        x = self.attention(inputs, inputs)
        x = self.dropout(x)
        _x = self.norm1(x + inputs)

        # Feed Forward Part
        x = self.conv1d1(_x)
        #x = self.liner1(_x)
        x = self.dropout(x)
        x = self.conv1d2(x)
        #x = self.liner2(x)
        x = self.norm2(x + _x)
        return x
   

# sequential model
model = keras.Sequential([
    keras.Input(shape=(timesteps, feature)),
    # transformer encoder layer
    # input shape is (batch_size, timesteps, features)
    TransformerEncoder(1, filters, feature, drop_rate),
    TransformerEncoder(1, filters, feature, drop_rate),

    keras.layers.GlobalAveragePooling1D(),
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
print("\nTransformer Test Accuracy: %.1f%%" % (100.0 * acc))