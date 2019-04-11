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

def shatter_img_list(img_list, out_file_list):
    """
    function to shatter img_list into list of lists.
    :param img_list: input list containing all image names for training or testing
    :param out_file_list: output file list of .tfrecords files
    :return: shattered_img_list: list of lists of filenames
    :return: out_file_list: append another one if there is more images left
    """
    shattered_img_list = []
    grb_idx = 0
    num_cases = len(img_list) // len(out_file_list)
    while True:
        if grb_idx + num_cases >= len(img_list):
            shattered_img_list.append(img_list[grb_idx:])
            break
        else:
            shattered_img_list.append(img_list[grb_idx:grb_idx + num_cases])
            grb_idx += num_cases
        print(grb_idx)
    if len(shattered_img_list) > len(out_file_list):
        out_file_list.append(out_file_list[-1] + 'rest')
    return shattered_img_list, out_file_list

img_list = ['img1', 'img2', 'img3', 'img4', 'img5']
out_list = ['out1', 'out2']

print(shatter_img_list(img_list, out_list))
print([(name + '.tfrecord') for name in img_list])
