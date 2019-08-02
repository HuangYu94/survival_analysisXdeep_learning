import tensorflow as tf
import numpy as np
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


input_risk = np.random.normal(size=(32,))
input_ph = tf.placeholder(dtype=tf.float32, shape=(32,))
input_survival = np.random.randint(1, 500, size=(32,))
survival_ph = tf.placeholder(dtype=tf.int32, shape=(32,))
input_censored = np.random.randint(0, 10, size=(32, ))>=7
censored_ph = tf.placeholder(dtype=tf.bool, shape=(32, ))


print(input_censored)

idx_lst = []
idx_left = []
idx_right = []
order_counter = 0
for i in range(0, 32):
    for j in range(0, 32):
        if not i==j:
            idx_lst.append([j, i])
            idx_left.append(j)
            idx_right.append(i)
            if input_survival[j] >= input_survival[i] and not input_censored[i]:
                order_counter += 1

print("Ordered count: %d"%order_counter)

paired_risk = tf.gather(input_ph, idx_lst)
risk_left = tf.gather(input_risk, idx_left)
risk_right = tf.gather(input_risk, idx_right)
mask = tf.cast(tf.gather(survival_ph, idx_left) >= tf.gather(survival_ph, idx_right), dtype=tf.float64) * (1-tf.cast(
    tf.gather(input_censored, idx_right), dtype=tf.float64))

edge_sum = tf.reduce_sum(mask)


print(risk_left.shape[0])

loss_log_sigmoid = (edge_sum + tf.reduce_sum(tf.log(tf.sigmoid(risk_left - risk_right)) * mask)/np.log(2))/edge_sum

loss_exp = (edge_sum - tf.reduce_sum(tf.exp(-(risk_left - risk_right)) * mask))/edge_sum

print(paired_risk)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
paired_val, loss_log_sigmoid_val, loss_exp_val, edge_sum_val = sess.run([paired_risk, loss_log_sigmoid, loss_exp, edge_sum], feed_dict={input_ph: input_risk, survival_ph: input_survival, censored_ph: input_censored})
print(edge_sum_val)
print(paired_val)
print(loss_log_sigmoid_val)
print(loss_exp_val)


