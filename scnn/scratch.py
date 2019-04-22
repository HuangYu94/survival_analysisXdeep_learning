# import tensorflow as tf
# import matplotlib.image as img
# import matplotlib.pyplot as plt
#
# image = img.imread('demo.jpg')
# reshaped_image = tf.cast(image,tf.float32)
# height = image.shape[0]//2
# width = image.shape[1]//2
# distorted_image = tf.random_crop(reshaped_image, [height, width, 3])
#
# print(distorted_image.shape)
# new_image = tf.cast(distorted_image,tf.uint8)
#
# with tf.Session() as sess:
#     plt.imshow(sess.run(new_image))
#     plt.show()

import numpy as np
from sklearn import preprocessing
X_train = np.random.normal(size=(3, 4))
print(X_train)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = min_max_scaler.fit_transform(X_train)
print(np.mean(X_train_scaled, axis=1))

