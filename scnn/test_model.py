import os
import math
import scnn.model_impl as model
import scnn.model_utils as utils
import time
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf

# define model parameters here:
batch_size = model.Batch_Size
# define test parameters here:
Aug_factor = 9


def eval_once(saved_model, saver, ckpt_path, test_logits, test_labels):
    """Run Eval once.

    Args:
      saver: Saver.
      summary_writer: Summary writer.
      top_k_op: Top K op.
      summary_op: Summary op.
    """
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(ckpt_path)
        #    current_model = int(saved_model)
        #    if ckpt and ckpt.model_checkpoint_path:
        if ckpt and ckpt.all_model_checkpoint_paths[saved_model]:
            # Restores from checkpoint
            #      saver.restore(sess, ckpt.model_checkpoint_path)
            saver.restore(sess, ckpt.all_model_checkpoint_paths[saved_model])
            # Assuming model_checkpoint_path looks something like:
            #   /my-favorite-path/cifar10_train/model.ckpt-0,
            # extract global_step from it.
            #      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            global_step = ckpt.all_model_checkpoint_paths[saved_model].split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))

            num_test_iter = int(math.ceil(NUM_EXAMPLES_PER_EPOCH_FOR_EVAL / args.bs))
            test_step = 0
            while test_step < num_test_iter and not coord.should_stop():
                print("\nTesting model: %s" % (saved_model + 1))
                print("Test batch: %s" % (test_step + 1))
                test_logits_value, test_labels_value = sess.run([test_logits, test_labels])
                if test_step == 0:
                    concatenated_test_logits = test_logits_value
                    concatenated_test_labels = test_labels_value['survival']
                    concatenated_test_observed = test_labels_value['observed']
                    concatenated_test_indexes = test_labels_value['idx']
                elif test_step >= 1:
                    concatenated_test_logits = np.concatenate((concatenated_test_logits, test_logits_value))
                    concatenated_test_labels = np.concatenate((concatenated_test_labels, test_labels_value['survival']))
                    concatenated_test_observed = np.concatenate(
                        (concatenated_test_observed, test_labels_value['observed']))
                    concatenated_test_indexes = np.concatenate((concatenated_test_indexes, test_labels_value['idx']))

                test_step += 1
                #        print('TEST Log_Likelihood is: %.4f'%(test_loss_value))
                model_tools.nptxt(test_logits_value, precision, os.path.join(args.r, 'TEST_batch_logits_all.txt'))
                model_tools.nptxt(test_labels_value['survival'], precision,
                                  os.path.join(args.r, 'TEST_batch_labels_all.txt'))
                model_tools.nptxt(test_labels_value['idx'], precision,
                                  os.path.join(args.r, 'TEST_batch_indexes_all.txt'))

            if test_step >= num_test_iter:
                # Calculate C-Index
                concatenated_test_logits, concatenated_test_labels, concatenated_test_observed, concatenated_test_at_risk, concatenated_test_indexes = eval_calc_at_risk(
                    concatenated_test_logits, concatenated_test_labels, concatenated_test_observed,
                    concatenated_test_logits.shape[0], concatenated_test_indexes)
                concatenated_test_score, concatenated_test_orderable = model_tools.c_index_score_orderable(
                    concatenated_test_logits, concatenated_test_labels, 1 - concatenated_test_observed)
                test_epoch_c_index = model_tools.c_index(concatenated_test_score, concatenated_test_orderable)
                model_tools.nptxt(test_epoch_c_index, precision,
                                  os.path.join(args.r, execution_time + '_TEST_epoch_cindex.txt'))

                # Saving the final index, predicted risk valiues, and survival times for test patients (each patient one risk)
                #          cnn_tools.nptxt(concatenated_test_logits, precision, os.path.join(FLAGS.results_dir,execution_time+'_TEST_patients_risks.txt'))
                #          cnn_tools.nptxt(concatenated_test_labels, precision, os.path.join(FLAGS.results_dir,execution_time+'_TEST_patients_survivals.txt'))
                #          cnn_tools.nptxt(concatenated_test_indexes, precision, os.path.join(FLAGS.results_dir,execution_time+'_TEST_patients_indexes.txt'))
                df = pd.DataFrame({"indexes": concatenated_test_indexes,
                                   "censored": 1 - concatenated_test_observed,
                                   "survivals": concatenated_test_labels,
                                   "risks": concatenated_test_logits[:, 0]})
                #          df_cindex = pd.DataFrame({"c-index": [test_epoch_c_index], "model": [ckpt.model_checkpoint_path], "randomization": [cnn_params.rand_num]})
                df_cindex = pd.DataFrame(
                    {"c-index": [test_epoch_c_index], "model": [ckpt.all_model_checkpoint_paths[saved_model]]})
                df[['indexes', 'censored', 'survivals', 'risks']].sort_values(by=['indexes'],
                                                                              ascending=True).reset_index(
                    drop=True).to_csv(os.path.join(args.r, execution_time + '_TEST_patients_results.csv'), sep='\t')
                df_cindex[['model', 'c-index']].to_csv(os.path.join(args.r, execution_time + '_TEST_models_cindex.csv'),
                                                       sep='\t')

            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='test_epoch_c_index @ 1', simple_value=test_epoch_c_index)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

        return df.sort_values(by=['indexes'], ascending=True).reset_index(drop=True)


def Evaluate(saved_model, tfrecord_iterator):
    with tf.Graph().as_default() as g:

        with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
            test_logits, test_labels = test_tower_loss(scope, tfrecord_iterator)

            # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.Moving_Average_Decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

    results_df = eval_once(saved_model, saver, test_logits, test_labels)

    return results_df


def image_augmentation(image, aug_factor, height, width, num_channels, batch_size):
    ''' This function gets:

    image: the original image which we want to do the test augmentation on it
    aug_factor: the augmentation factor which is the number of randomely cropped
    patches out of the image
    height: desired height for each cropped image
    width: desired width for each cropped image
    num_channels: number of channels for the original image

    and returns the randomely cropped image patches in a list. '''

    aug_images = {}

    for i in range(aug_factor):
        aug_images[i] = tf.random_crop(image, [batch_size, height, width, num_channels])

    return aug_images


def test_augmentation(images_dict):
    for i in range(len(images_dict)):
        # if experiment == 'image_genome':
        #     test_labels['genomics'] = tf.cast(test_labels['genomics'], tf.float32)
        if i == 0:
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                test_logit = model.inference(images_dict[i], model.Keep_Prob, batch_size)
        elif i != 0:
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                test_logit = model.inference(images_dict[i], model.Keep_Prob, batch_size)
        if i == 0:
            test_logits = test_logit
        else:
            test_logits = tf.concat_v2([test_logits, test_logit], 1)
    test_logits, test_orders = tf.nn.top_k(test_logits, k=len(images_dict), sorted=True, name=None)
    if len(images_dict) % 2.0 == 0:
        augmented_logits = tf.reshape((test_logits[:, tf.cast(len(images_dict) / 2 - 1, tf.int32)] +
                                       test_logits[:, tf.cast(len(images_dict)/2,tf.int32)]) / 2.0,(-1, 1))
    else:
        augmented_logits = tf.reshape(test_logits[:, tf.cast((len(images_dict) - 1) / 2, tf.int32)], (-1, 1))

    return augmented_logits


def test_tower_loss(scope, tfrecord_iterator):

    survival_months, censored, patient_id, histology_image = tfrecord_iterator.get_next()

    test_images, test_labels = model.calc_at_risk(histology_image, survival_months, censored)
    aug_images = image_augmentation(test_images, Aug_factor, model.Crop_Size,
                                    model.Crop_Size, model.Num_of_Channels_Image, batch_size)
    test_logits = test_augmentation(aug_images)

    # "test_labels" is a dictionary in which the keywords are among: 'idx'
    # for patients' indexes, 'survival', 'censored', 'observed', 'idh',
    # 'codel', 'copynum' for copy numbers, 'at_risk'
    return test_logits, test_labels



num_models = model.Num_of_Model_Averaging
all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_test_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test')
all_df = pd.read_csv(all_data_csv_path)
test_image_list = os.listdir(all_test_images_path)

ReCreate_Record = True
num_record_split = 3
# convert test data into .tfrecords
out_file_names = [('test_image_record_file' + str(i)) for i in range(0, num_record_split)]
tfrecords_file_name = [os.path.join(os.path.dirname(__file__), out_file_name) for out_file_name in out_file_names]
some_train_img_list = []
for img_str in test_image_list:
    some_train_img_list.append(os.path.join(all_test_images_path, img_str))

if ReCreate_Record:
    utils.CreateTFRecords(some_train_img_list, out_file_names, all_df)


tfrecord_iterator = utils.DecodeTFRecords(out_file_names, batch_size)


model_results = {}
for saved_model in range(0, num_models):



