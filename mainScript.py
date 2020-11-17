import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from KNN import KNN

# import the training and testing data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# show an image
# plt.imshow(x_train[1,:,:,:])

# classify images
K = 5  # neighbors
D = 1  # city block
Errors = 0
for k in range(0, 2):

    Yhat = KNN(x_test[k, ...], x_train[0:5, :, :, :], y_train[0:5], K, D)

    #if (Yhat != y_test[k]):
     #   Errors += 1
