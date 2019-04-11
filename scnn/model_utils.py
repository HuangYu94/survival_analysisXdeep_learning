import os
import scnn.model_impl as model

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

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

def CreateTFRecords(img_file_list, out_file_names, csv_features):
    """
    Given image file list, this function grab image files and labels(survival months, censoring status) to create
    TFRecords files on disk.
    :param img_file_list: input file directories list containing directory strings
    :param out_file_names: list of output file names since we need to store in multiple .tfrecords files
    :param csv_features: pandas dataframe containing other information(patient ID, survival info, censoring status,
    genomic features etc...)
    :return: None.
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
        if len(shattered_img_list) > len(out_file_list):
            out_file_list.append(out_file_list[-1] + 'rest')
        return shattered_img_list, out_file_list

    img_file_lists, out_file_names = shatter_img_list(img_file_list, out_file_names)
    for img_file_idx, img_file_list in enumerate(img_file_lists):
        out_file_name = out_file_names[img_file_idx]
        with tf.python_io.TFRecordWriter(out_file_name + '.tfrecords') as writer:
            for img_name_path in img_file_list:
                patient_id = _get_imgID(img_name_path)
                histology_image = _get_rgb(img_name_path)
                # print(patient_id)
                # print(csv_features[csv_features['TCGA ID'] == patient_id]['Survival months'].as_matrix())
                survival_month = csv_features[csv_features['TCGA ID'] == patient_id]['Survival months'].as_matrix()[0]
                censored = csv_features[csv_features['TCGA ID'] == patient_id]['censored'].as_matrix()[0]
                serialized_feature = serialize_example(survival_month, censored, patient_id.encode('utf8'),
                                                       histology_image.tobytes())
                writer.write(serialized_feature)


def DecodeTFRecords(records_file_name_queue, batch_size = 64, img_hei = 1024, img_wid = 1024, img_channel = 3):
    """
    function to read and decode TFRecords file name queue
    :param records_file_name_queue: TFRecords file name queue
    :return: tfrecord_iterator: stores all data as a tfrecord_iterator
    """
    records_file_name_queue = [(name + '.tfrecords') for name in records_file_name_queue]
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
    dataset = dataset.map(_parse_, num_parallel_calls=batch_size).batch(batch_size, drop_remainder=True)
    dataset = dataset.shuffle(buffer_size=100, reshuffle_each_iteration=True).repeat()
    tfrecord_iterator = dataset.make_initializable_iterator()
    return tfrecord_iterator

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


def Test_Utils():
    num_batches = model.num_batches
    batch_size = model.Batch_Size

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

    ReCreate_Record = False
    num_records_split = 5
    out_file_names = [('train_image_record_file' + str(i)) for i in range(0, num_records_split)]
    tfrecords_file_name = [os.path.join(os.path.dirname(__file__), out_file_name) for out_file_name in out_file_names]

    some_train_img_list = []
    for img_str in train_image_list:
        some_train_img_list.append(os.path.join(all_train_images_path, img_str))

    # for out_file_name in out_file_names:
    #     tfrecords_file_name = os.path.join(os.path.dirname(__file__), out_file_name)


    if ReCreate_Record:
        CreateTFRecords(some_train_img_list, out_file_names, all_df)

    filename_queue = tf.train.string_input_producer(out_file_names)
    tfrecord_iterator = DecodeTFRecords(out_file_names, batch_size)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    with tf.Session() as sess:
        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(0, model.Max_Num_Epoch):
            sess.run(tfrecord_iterator.initializer)
            print("current i is %d and initilizer called"%(i))
            survival_months, censored, patient_ID, histology_image = sess.run(tfrecord_iterator.get_next())
            if i == 3:
                print(survival_months)
                print(censored)
                print(patient_ID)
                plt.imshow(histology_image[1, :, :, :])
                plt.show()
        coord.request_stop()
                # raise ValueError("STOP HERE!")
            # while True:
            #     try:
            #         survival_months, censored, patient_ID, histology_image = sess.run(tfrecord_iterator.get_next())
            #         print("current i is %d and initilizer called" % (i))

            #     except:
            #         break


