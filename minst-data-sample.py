import keras
import numpy as np
import matplotlib.pyplot as plt

# load the dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
assert x_train.shape == (60000, 28, 28)
assert x_test.shape == (10000, 28, 28)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)

#print(x_train[0], x_train[0].shape, y_train[0])

# sample 25 mnist digits from train dataset
indexes = np.random.randint(0, x_train.shape[0], size=25)
images = x_train[indexes]
labels = y_train[indexes]

# plot the 25 mnist digits and label
plt.figure(figsize=(10,10))
for i in range(len(indexes)):
    plt.subplot(5, 5, i + 1)
    plt.title(labels[i])
    image = images[i]
    plt.imshow(image, cmap='gray')
    plt.axis('off')
 
plt.savefig("./tmp/mnist-samples.png")
plt.show()
plt.close('all')