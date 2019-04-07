import tensorflow as tf
import matplotlib.image as img
import matplotlib.pyplot as plt

image = img.imread('demo.jpg')
reshaped_image = tf.cast(image,tf.float32)
height = image.shape[0]//2
width = image.shape[1]//2
distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

print(distorted_image.shape)
new_image = tf.cast(distorted_image,tf.uint8)

with tf.Session() as sess:
    plt.imshow(sess.run(new_image))
    plt.show()

