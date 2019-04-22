import os
import scnn.model_impl as model
import scnn.model_utils as utils
import time
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf



num_batches = model.num_batches
batch_size = model.Batch_Size

all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_train_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'train')
all_df = pd.read_csv(all_data_csv_path)
num_cases, _ = all_df.shape
num_all_batches = num_cases // batch_size + 1
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

if ReCreate_Record:
    utils.CreateTFRecords(some_train_img_list, out_file_names, all_df)

_, out_file_names = utils.shatter_img_list(some_train_img_list, out_file_names)
print(out_file_names)
parsed_example_batch = utils.DecodeTFRecords(out_file_names, batch_size)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

# str_list = []
# train_patient_id = map(utils.get_imgID, train_image_list)
# str_set = set()
# id_set = set()
#
#
# with tf.Session() as sess:
#     sess.run(init_op)
#     coord = tf.train.Coordinator()
#     threads = tf.train.start_queue_runners(coord=coord)
#     # sess.run(tfrecord_iterator.initializer)
#
#     for i in range(0, num_all_batches + 100):
#         patient_id = sess.run(parsed_example_batch['patient ID'])
#         for id in patient_id:
#             id_set.add(id)
#
#     coord.request_stop()
#     coord.join(threads)
#
# print(len(id_set))
# TRAIN FUNCTION STARTED HERE!!
with tf.device('/cpu:0'):
    # tf.set_random_seed(model_params.seed)
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = len(train_image_list) // model.Batch_Size
    decay_steps = int(num_batches_per_epoch * model.Num_of_Epochs_Per_Decay)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(model.Learning_Rate,
                                    global_step,
                                    decay_steps,
                                    model.Learing_Rate_Decay_Factor,
                                    staircase=True)


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
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            loss, logits, labels, images = model.tower_loss(scope, parsed_example_batch, batch_size)

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
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=model.Num_of_Model_Averaging)

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

    # all_features.to_csv(os.path.join(args.r, execution_time + '_all_features.csv'), sep='\t')
    for epoch in range(1, model.Max_Num_Epoch + 1):
        # if epoch == args.me:
        #     shutil.rmtree(args.d, ignore_errors=True)
        #     break

        #      if step != 0:
        step = 0
        epoch_loss_value = 0
        batch_num = 0
        precision = 5
        for _ in range(0, model.num_steps_per_train_epoch):
            start_time = time.time()

            _, loss_value, logits_value, labels_value, images_value = sess.run([train_op, loss, logits, labels, images])

            duration = time.time() - start_time

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
            step += 1
            epoch_loss_value = epoch_loss_value + loss_value * batch_size
        utils.LogDataAsText(epoch_loss_value, precision, model.train_log_folder, model.epoch_loss_fname)
        if epoch > (model.Max_Num_Epoch - model.Num_of_Model_Averaging):
            # save last Num_of_Model_Averaging models for testing
            train_ckpt_folder = os.path.join(os.path.dirname(__file__), model.train_ckpt_folder)
            if not os.path.exists(train_ckpt_folder):
                os.mkdir(train_ckpt_folder)
            ckpt_path = os.path.join(train_ckpt_folder, 'model.ckpt')
            saver.save(sess, ckpt_path, global_step=epoch)




