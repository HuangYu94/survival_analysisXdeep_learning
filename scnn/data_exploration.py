import os
import scnn.model_impl as model
import time
from datetime import datetime
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
    :param csv_features: pandas dataframe containing other information(patient ID, survival info, censoring status,
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

out_flie_name = 'some_batches.tfrecords'
some_train_img_list = []
for img_str in train_image_list[0:num_batches*batch_size]:
    some_train_img_list.append(os.path.join(all_train_images_path, img_str))

tfrecords_file_name = os.path.join(os.path.dirname(__file__), out_flie_name)
if not os.path.exists(tfrecords_file_name):
    CreateTFRecords(some_train_img_list, out_flie_name, all_df)

filename_queue = tf.train.string_input_producer([out_flie_name], num_epochs=3)
tfrecord_iterator = DecodeTFRecords(out_flie_name, batch_size)

# init_op = tf.group(tf.global_variables_initializer(),
#                    tf.local_variables_initializer())
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     for i in range(0, num_batches):
#         survival_months, censored, patient_ID, histology_image = sess.run(tfrecord_iterator.get_next())
#         print("current i is %d"%(i))
#         if i == 3:
#             print(survival_months)
#             print(censored)
#             print(patient_ID)
#             plt.imshow(histology_image[1, :, :, :])
#             plt.show()

with tf.device('/cpu:0'):
    # tf.set_random_seed(model_params.seed)
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    # num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
    #                          args.bs)
    # decay_steps = int(num_batches_per_epoch * model_params.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    # lr = tf.train.exponential_decay(args.lr,
    #                                 global_step,
    #                                 decay_steps,
    #                                 model_params.LEARNING_RATE_DECAY_FACTOR,
    #                                 staircase=True)

    lr = 0.0001

    # Create an optimizer that performs gradient descent.
    opt = model.optimizer(lr)[model.Optimizer_Name]

    # Calculate the gradients for each model tower.
    tower_grads = []

    with tf.name_scope('%s_%d' % (model.Tower_Name, 0)) as scope:
        # Calculate the loss for one tower of the cnn model. This function
        # constructs the entire cnn model but shares the variables across
        # all towers.
        #          loss, logits, labels, images, observed, indexes, copy_numbers = tower_loss(scope)

        # Reuse variables for the next tower.
        # tf.get_variable_scope().reuse_variables()
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            loss, logits, labels, images = model.tower_loss(scope, tfrecord_iterator, batch_size)

            # Retain the summaries from the final tower.
            # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

            # Calculate the gradients for the batch of data on this cnn tower.
            grads = opt.compute_gradients(loss)

            # Keep track of the gradients across all towers.
            tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = model.average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    # summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    # for grad, var in grads:
    #     if grad is not None:
    #         summaries.append(
    #             tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    # for var in tf.trainable_variables():
    #     summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        model.Moving_Average_Decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    # saver = tf.train.Saver(tf.global_variables(), max_to_keep=Num_of_Model_Averaging)

    # Build the summary operation from the last tower summaries.
    # summary_op = tf.summary.merge(summaries)

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    # summary_writer = tf.summary.FileWriter(args.t,
    #                                        graph=sess.graph)
    epoch = 0
    epoch_loss_value = 0
    batch_num = 0
    precision = 5

    # all_features.to_csv(os.path.join(args.r, execution_time + '_all_features.csv'), sep='\t')
    for step in range(1, model.Max_Steps + 1):
        # if epoch == args.me:
        #     shutil.rmtree(args.d, ignore_errors=True)
        #     break

        #      if step != 0:
        start_time = time.time()
        _, loss_value, logits_value, labels_value, images_value = sess.run([train_op, loss, logits, labels, images])
        duration = time.time() - start_time
        #      batch_num = batch_num + 1

        #      if step % 1 == 0:
        num_examples_per_step = model.Batch_Size * model.Num_GPUs
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = duration / model.Num_GPUs
        batch_num = batch_num + 1
        print('\nTraining epoch: %s' % (epoch + 1))
        print('Batch number: %s' % batch_num)
        format_str = ('%s: step %d, Train Log_Likelihood = %.4f (%.1f HPFs/sec; %.3f '
                      'sec/batch)')
        print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        epoch_loss_value = epoch_loss_value + loss_value * batch_size


