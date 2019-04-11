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

import os


def LogDataAsText(data, precision, dirname, fname):
    """
    function to log data(like losses) created during training as text file
    :param data: data to be logged could be string number etc.
    :param precision: precision for float
    :param dirname: log directory name
    :param fname: log file name
    :return: None
    """
    import numpy as np
    import os

    if not os.path.exists(dirname):
        os.mkdir(dirname)
    with open(os.path.join(dirname, fname), 'a') as f:
        if type(data) == type('string'):
            np.savetxt(f, np.array([data]), fmt='%s')
        elif type(data) == type(np.array(['string'])):
            np.savetxt(f, data, fmt='%s')
        elif type(data) == type(1) or type(data) == type(1.0) or type(data) == type(np.float32(1)) or type(
                data) == type(np.float64(1)) or type(data) == type(np.float(1)):
            np.savetxt(f, np.array([data]), fmt='%.' + '%d' % precision + 'f')
        elif type(data) == type(np.array([1])) or type(data) == type(np.array([1.0])):
            np.savetxt(f, data, fmt='%.' + '%d' % precision + 'f')
        f.close()


precision = 5
data = 1.2333
dirname = os.path.join(os.path.dirname(__file__), 'train_log')
fname = 'epoch_loss.txt'
LogDataAsText(data, precision, dirname, fname)
data1 = 0.9999
LogDataAsText(data1, precision, dirname, fname)

