import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def NormalizeRGB(img_in):
    """
    function to normalize RGB to float [0,1]
    :param img_in: image uint8
    :return: normalized float image in range [0,1]
    """
    img_float = np.asarray(img_in, dtype=np.double)
    hei, wid, channel = img_float.shape
    img_out = np.zeros((hei, wid, 3))
    for i in range(0, 3):
        img_out[:, :, i] = (img_float[:, :, i] - np.min(img_float[:, :, i])) / np.max(img_float[:, :, i])
    return img_out

def CreateTFRecords(img_file_list, out_file_name, csv_features):
    """
    Given image file list, this function grab image files and labels(survival months, censoring status) to create
    TFRecords file on disk.
    :param img_file_list: input file directories list containing directory strings
    :param out_file_name: output file name
    :param csv_features: pandas dataframe containing other information(patient ID, survival info, cencering status,
    genomic features etc...)
    :return: None
    """
    def _bytes_feature(value):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _float_feature(value):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    def _int64_feature(value):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def _get_rgb(image_in):
        """
        just grab channel 0 to 3
        :param image_in:
        :return:
        """
        img_PIL = Image.open(image_in)
        return np.asarray(img_PIL, dtype=np.uint8)[:, :, 0:3]

    def serialize_example(feature0, feature1, feature2, feature3):
        """
        Creates a tf.Example message ready to be written to a file.
        """

        feature = {
            'survival months': _int64_feature(feature0),
            'censored': _int64_feature(feature1),
            'patient ID': _bytes_feature(feature2),
            'histology image': _bytes_feature(feature3),
        }

        # Create a Features message using tf.train.Example.
        example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
        return example_proto.SerializeToString()
    def _get_imgID(img_path_name):
        img_id_start = img_path_name.find('TCGA')
        return img_path_name[img_id_start:img_id_start+12]
    survival_months = []
    censored = []
    patient_ID = []
    histology_image = []
    with tf.python_io.TFRecordWriter(out_file_name) as writer:
        for i, img_name_path in enumerate(img_file_list):
            img_ID = _get_imgID(img_name_path)
            img_rgb = _get_rgb(img_name_path)
            histology_image.append(img_rgb)
            # print(img_ID)
            # print(csv_features[csv_features['TCGA ID'] == img_ID]['Survival months'].as_matrix())
            survival_months.append(csv_features[csv_features['TCGA ID'] == img_ID]['Survival months'].as_matrix()[0])
            censored.append(csv_features[csv_features['TCGA ID'] == img_ID]['censored'].as_matrix()[0])
            patient_ID.append(img_ID)
            serialized_feature = serialize_example(survival_months[i], censored[i], patient_ID[i].encode('utf8'),
                                                   histology_image[i].tobytes())
            writer.write(serialized_feature)

def DecodeTFRecords(records_file_name_queue, batch_size = 64, img_hei = 1024, img_wid = 1024, img_channel = 3):
    """
    function to read and decode TFRecords file name queue
    :param records_file_name_queue: TFRecords file name queue
    :return: tfrecord_iterator: stores all data as a tfrecord_iterator
    """
    def _parse_(serialized_example):
        feature = {'survival months': tf.FixedLenFeature([], tf.int64),
                   'censored': tf.FixedLenFeature([], tf.int64),
                   'patient ID': tf.FixedLenFeature([], tf.string),
                   'histology image': tf.FixedLenFeature([], tf.string)}
        example = tf.parse_single_example(serialized_example, feature)
        histo_image = tf.decode_raw(example['histology image'], tf.uint8)
        histo_image = tf.reshape(histo_image, [img_hei, img_wid, img_channel])
        return example['survival months'], example['censored'], example['patient ID'], histo_image
    dataset = tf.data.TFRecordDataset(records_file_name_queue)
    dataset = dataset.map(_parse_, num_parallel_calls=batch_size).batch(batch_size)
    tfrecord_iterator = dataset.make_one_shot_iterator()
    return tfrecord_iterator





num_batches = 5
batch_size = 64

all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_train_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'train')
all_df = pd.read_csv(all_data_csv_path)
train_image_list = os.listdir(all_train_images_path)
# sess = tf.InteractiveSession()
first_image_path = os.path.join(all_train_images_path, train_image_list[0])
# print(first_image_path)
# one_image = tf.image.decode_image(tf.read_file(first_image_path))
# print(one_image.eval().shape)

# one_image = Image.open(first_image_path)
# one_image = NormalizeRGB(one_image)
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.imshow(one_image[:, :, 0:3])
# plt.show()

out_flie_name = 'some_batches.tfrecords'
some_train_img_list = []
for img_str in train_image_list[0:num_batches*batch_size]:
    some_train_img_list.append(os.path.join(all_train_images_path, img_str))

print(some_train_img_list)
CreateTFRecords(some_train_img_list, out_flie_name, all_df)

filename_queue = tf.train.string_input_producer([out_flie_name], num_epochs=3)
tfrecord_iterator = DecodeTFRecords(out_flie_name)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    for i in range(0, num_batches):
        survival_months, censored, patient_ID, histology_image = sess.run(tfrecord_iterator.get_next())
        print("current i is %d"%(i))
        if i == 3:
            print(survival_months)
            print(censored)
            print(patient_ID)
            plt.imshow(histology_image[1, :, :, :])
            plt.show()






