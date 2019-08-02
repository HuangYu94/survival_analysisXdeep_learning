#!/usr/bin/env python2
"""

@author: Pooya Mobadersany

Please cite the following paper for any use:
    
Mobadersany, Pooya, et al. "Predicting cancer outcomes from histology and genomics using convolutional networks." bioRxiv (2017): 198010.

Find paper at: https://goo.gl/5ff2Wk

"""

from __future__ import absolute_import, division, print_function
from datetime import datetime
import os.path
import re
import time
import numpy as np
from six.moves import xrange
import tensorflow as tf
import model
import model_input
import model_tools
import model_params
import shutil
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='Arguments for Training the SCNN/GSCNN model')

parser.add_argument('-m', type=str, default='image_only',
                    help='scnn or gscnn; scnn for image_only and gscnn for image_genome')

parser.add_argument('-p', type=str, default='./inputs/ImageSplits.txt',
                    help='Path to the Train/Test patient list')

parser.add_argument('-f', type=str, default='./inputs/all_dataset.csv',
                    help='Path to the features data')

parser.add_argument('-i', type=str, default='../images/train',
                    help='Path to the Train set ROIs')

parser.add_argument('-r', type=str, default='./train_results',
                    help='Path to the Training results')

parser.add_argument('-t', type=str, default='./checkpoints',
                    help='Path to the Training meta files')

parser.add_argument('-d', type=str, default='./tmp',
                    help='Path to the temporary binary files for Training')

parser.add_argument('--lr', type=float, default=0.0002,
                    help='Initial learning rate')

parser.add_argument('--me', type=int, default=100,
                    help='Max number of epochs')

parser.add_argument('--kp', type=float, default=0.95,
                    help='Keeping probability for Train time dropout')

parser.add_argument('--bs', type=int, default=14,
                    help='batch size')

parser.add_argument('--ic', type=int, default=1,
                    help='indexes column in the all_dataset.csv file (Starting from 0)')

parser.add_argument('--ngf', type=int, default=2,
                    help='Number of genomic features. (e.g. 2 when we have: indexes, censored, Survival months, codeletion, idh mutation)')

parser.add_argument('--gn', type=str, default='n',
                    help='Z Normalization of the geomic values; y for yes and n for no')

parser.add_argument('--ns', type=int, default=6,
                    help='Normalization Starting column number for the geomic values we want to do the Z normalization (Starting from 0); Note: The columns we want to normalize should be at the last columns of the all_data.csv file')

parser.add_argument('--nm', type=int, default=5,
                    help='number of models for model averaging in test time, we want last 5 models (96-100) so the default is 5.')


args = parser.parse_args()
FLAGS = tf.app.flags.FLAGS

global execution_time
execution_time = time.strftime("%Y%m%d_%H%M")

print('\n\nexperiment is: %s'%args.m)

#########################
### Defining Features ###
#genome_columns = {}
#genome_columns['start'] = 4 # number of first column for the desired genomic data in the .csv (e.g. Extended_GSCNN_Dataset_normalized_copy_numbers_FixedIndexes.csv) file
#genome_columns['end'] = 84 # number of last column for the desired genomic data in the .csv (e.g. Extended_GSCNN_Dataset_normalized_copy_numbers_FixedIndexes.csv) file
#features_path = '/home/pooya/Desktop/Spring_2017/Group_Survival/Final_Experiments/CLEANED_SCNN_Models_25_Oct_2017/image_only_image_genome_EXTRA_GENOME/outputs/Extended_GSCNN_Dataset_normalized_copy_numbers.csv'
#features_path = './outputs/all_dataset.csv'
features_path = args.f
# read the all_dataset.csv file as a pandas data frame:
features = pd.read_csv(features_path, header=0)
# Get the 'TCGA ID' of the patients from data fram:
TCGA_IDs = features.iloc[:, 0]
# get the indexes, censored, Survival months:
all_features_org = features.iloc[:, args.ic:args.ic+3]
#if experiment == 'image_only':
# TCGA ID, indexes, censored, Survival months
all_features_org = pd.concat([TCGA_IDs, all_features_org], axis=1)
if args.m == 'image_genome':
    genomics = features.iloc[:, args.ic+3:args.ic+3+args.ngf] # copy_numbers are from column #14 (EGFR) to the end.
    if args.gn == 'y':
        if args.ns >= args.ic+3+args.ngf:
			raise ValueError('The starting column number for the normalization of the genomic features is not valid! It should be a smaller number. Define with "--ns"')
        not_normalized_genomics = features.iloc[:, args.ic+3:args.ns]
        normalized_genomics_tmp = features.iloc[:, args.ns:]
        normalized_genomics = model_tools.z_normalization(normalized_genomics_tmp)
        genomics = pd.concat([not_normalized_genomics, normalized_genomics], axis=1)
    all_features_org = pd.concat([all_features_org, genomics], axis=1)
# find the index of the rows with NaN values in all_features data frame
nan_indexes = pd.isnull(all_features_org).any(1).nonzero()[0]
# remove rows with NaN values from all_features data frame:
all_features = all_features_org.drop(all_features_org.index[nan_indexes])
available_patients = all_features['TCGA ID'].tolist() # List of all the patients that have "not_NaN" features
number_of_labels = len(all_features.columns)-1 # number of labels in each binary file naming format: |Idh|Codeletion|Survival|censored|index|image|
#########################

data = model_tools.data_generator(args.i, available_patients) # A dictionary containing the list of images in Train set (data['train']) and Test set (data['test'])
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = len(data)

if args.m == 'image_only':
    trainset_base_name = args.m+'_idx-censored-survival-image-GBMLGG_20x_'+str(model_params.org_image['height'])+'x'+str(model_params.org_image['width'])+'x'+str(model_params.org_image['channels'])+'_float16_train_'
    train_chunk_name = trainset_base_name+'%d.bin'

elif args.m == 'image_genome':
    trainset_base_name = args.m+'_idx-censored-survival-genomics-image-GBMLGG_20x_'+str(model_params.org_image['height'])+'x'+str(model_params.org_image['width'])+'x'+str(model_params.org_image['channels'])+'_float16_train_'
    train_chunk_name = trainset_base_name+'%d.bin'

# Creating the binary files for train set if they don't exist
if not os.path.isfile(os.path.join(args.d, train_chunk_name%1)):
    class params:
        experiment = args.m # 'image_only' or 'image_genome'
        DataType = 'train'
        train_chunks = model_params.train_chunks # number of generated binary files for train set
    # destination for produced binary dataset files
    if not os.path.exists(args.d):
        os.makedirs(args.d)
    if model_params.org_image['channels'] == 1:
        model_tools.MakeBin(params, data, all_features, args.d, trainset_base_name, args.i, ext='png', label_range=np.float16, DesiredChannelMode='G') # to pickle data in float16 format (label_range=np.float16) into a binary file
    elif model_params.org_image['channels'] == 3:
        model_tools.MakeBin(params, data, all_features, args.d, trainset_base_name, args.i, ext='png', label_bytes=np.float16, DesiredChannelMode='RGB') # to pickle data in float16 format (label_range=np.float16) into a binary file
    else:
        raise ValueError('Channel Number of org_image is invalid, fix it in model_params.py file!')
    
num_steps_per_train_epoch = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/args.bs) # 1239/14 = int(88.5) = 88
#max_steps = int(num_steps_per_train_epoch*(args.me+1)) # Number of batches to run = 88*(5+1) = 528
max_steps = int(num_steps_per_train_epoch*args.me) # Number of batches to run = 88*(5) = 440


def calc_at_risk(X, Y, experiment):
   """
   Calculate the at risk group of all patients. For every patient i, this
   function returns the index of the first patient who died after i, after
   sorting the patients w.r.t. time of death.
   Refer to the definition of
   Cox proportional hazards log likelihood for details: https://goo.gl/k4TsEM
     
   Parameters
   ----------
   X: numpy.ndarray
   m*n matrix of expression data
   survivals: numpy.ndarray
   m sized vector of time of death
   observed: numpy.ndarray
   indexes: numpy.ndarray of patients' indexes
   m sized vector of observed status (1 - censoring status)

   Returns
   -------
   X: numpy.ndarray
   m*n matrix of expression data sorted w.r.t time of death
   survivals: numpy.ndarray
   m sized sorted vector of time of death
   observed: numpy.ndarray
   m sized vector of observed status sorted w.r.t time of death
   at_risk: numpy.ndarray
   indexes: numpy.ndarray
   m sized vector of starting index of risk groups
   """

   values, order = tf.nn.top_k(Y['survival'], args.bs, sorted=True)
   values = tf.reverse(values, axis=[0])
   order = tf.reverse(order, axis=[0])
   sorted_survivals = values   
   Y['at_risk'] = tf.nn.top_k(sorted_survivals, args.bs, sorted=False).indices
   Y['survival'] = sorted_survivals
   Y['observed'] = tf.gather(Y['observed'], order)
   X = tf.gather(X, order)
   Y['idx'] = tf.gather(Y['idx'], order)
   if experiment == 'image_genome':
       Y['genomics'] = tf.gather(Y['genomics'], order)
   
   return X, Y
   
   
def tower_loss(scope):
  """Calculate the total loss on a single tower running the cnn model.
  Args:
    scope: unique prefix string identifying the cnn tower, e.g. 'tower_0'
  Returns:
     Tensor of shape [] containing the total loss for a batch of data
  """
  images, labels = model.distorted_inputs(args.m, train_chunk_name, args.bs, number_of_labels, NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN, args.d)
  labels['observed'] = 1 - labels['censored']
  # images, labels = calc_at_risk(images, labels, args.m)
  if args.m == 'image_genome':
      labels['genomics'] = tf.cast(labels['genomics'], tf.float32)

  logits = model.inference(images, labels, args.kp, args.m, args.bs)
  _, risk_diff = model.log_sigmoid_loss(logits, labels, args.bs)
  losses = tf.get_collection('losses', scope)
  total_loss = tf.add_n(losses, name='total_loss')
  for l in losses + [total_loss]:
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
    tf.summary.scalar(loss_name, l)
    
  # "labels" is a dictionary in which the keywords are among: 'idx' for
  # patients' indexes, 'survival', 'censored', 'observed', 'idh', 'codel',
  # 'copynum' for copy numbers, 'at_risk'
  return total_loss, logits, labels, images, risk_diff


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(grads, 0)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def train():
  """Train cnn for a number of steps."""
  with tf.Graph().as_default(), tf.device('/cpu:0'):
    tf.set_random_seed(model_params.seed)
    # Create a variable to count the number of train() calls. This equals the
    # number of batches processed * FLAGS.num_gpus.
    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    # Calculate the learning rate schedule.
    num_batches_per_epoch = (NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN /
                             args.bs)
    decay_steps = int(num_batches_per_epoch * model_params.NUM_EPOCHS_PER_DECAY)

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(args.lr,
                                    global_step,
                                    decay_steps,
                                    model_params.LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    # Create an optimizer that performs gradient descent.
    opt = model_input.optimizer(lr)[model_params.optimizer]

    # Calculate the gradients for each model tower.
    tower_grads = []
    for i in xrange(FLAGS.num_gpus):
      with tf.device('/gpu:%d' % i):
        with tf.name_scope('%s_%d' % (model.TOWER_NAME, i)) as scope:
          # Calculate the loss for one tower of the cnn model. This function
          # constructs the entire cnn model but shares the variables across
          # all towers.
#          loss, logits, labels, images, observed, indexes, copy_numbers = tower_loss(scope)
          loss, logits, labels, images, risk_diff = tower_loss(scope)

          # Reuse variables for the next tower.
          with tf.variable_scope(tf.get_variable_scope(), reuse=None):

              # Retain the summaries from the final tower.
              summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

              # Calculate the gradients for the batch of data on this cnn tower.
              grads = opt.compute_gradients(loss)

              # Keep track of the gradients across all towers.
              tower_grads.append(grads)

    # We must calculate the mean of each gradient. Note that this is the
    # synchronization point across all towers.
    grads = average_gradients(tower_grads)

    # Add a summary to track the learning rate.
    summaries.append(tf.summary.scalar('learning_rate', lr))

    # Add histograms for gradients.
    for grad, var in grads:
      if grad is not None:
        summaries.append(
            tf.summary.histogram(var.op.name + '/gradients', grad))

    # Apply the gradients to adjust the shared variables.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    # Add histograms for trainable variables.
    for var in tf.trainable_variables():
      summaries.append(tf.summary.histogram(var.op.name, var))

    # Track the moving averages of all trainable variables.
    variable_averages = tf.train.ExponentialMovingAverage(
        model_params.MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    # Group all updates to into a single train op.
    train_op = tf.group(apply_gradient_op, variables_averages_op)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=args.nm)

    # Build the summary operation from the last tower summaries.
    summary_op = tf.summary.merge(summaries)    

    # Build an initialization operation to run below.
    init = tf.global_variables_initializer()

    # Start running operations on the Graph. allow_soft_placement must be set to
    # True to build towers on GPU, as some of the ops do not have GPU
    # implementations.
    sess = tf.Session(config=tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=FLAGS.log_device_placement))
    sess.run(init)

    # Start the queue runners.
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(args.t,
                                            graph=sess.graph)
    epoch = 0
    epoch_loss_value = 0     
    batch_num = 0 
    precision = 5                            
 
    all_features.to_csv(os.path.join(args.r,execution_time+'_all_features.csv'), sep='\t')
    epoch_CI = 0
    for step in xrange(1, max_steps+1):
      if epoch == args.me:
	shutil.rmtree(args.d, ignore_errors=True)
	break
          
#      if step != 0:
      # check number of observed:

      start_time = time.time()
      _, loss_value, logits_value, labels_value, images_value, risk_diff_val = sess.run([train_op, loss, logits, labels, images, risk_diff])
      duration = time.time() - start_time
#      batch_num = batch_num + 1

#      if step % 1 == 0:
      num_examples_per_step = args.bs * FLAGS.num_gpus
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = duration / FLAGS.num_gpus
      batch_num = batch_num + 1
      batch_score, batch_orderable = model_tools.c_index_score_orderable(logits_value, labels_value['survival'], labels_value['censored'])
      batch_CI = model_tools.c_index(batch_score, batch_orderable)
      # print(risk_diff_val)
      # print(risk_diff_val.shape)
      print('\nTraining epoch: %s'%(epoch+1))
      print('Batch number: %s'%batch_num)
      format_str = ('%s: step %d, Train Log_Likelihood = %.4f (%.1f HPFs/sec; %.3f '
                    'sec/batch)')
      print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
      print("The batch C index is: %1.4f number of observed %1.4f"%(batch_CI, args.bs - np.sum(labels_value['censored'])))
      epoch_CI += batch_CI
      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      epoch_loss_value = epoch_loss_value + loss_value*args.bs                
      if step % num_steps_per_train_epoch == 0:
          epoch_loss_value = epoch_loss_value/(num_steps_per_train_epoch*args.bs)
          model_tools.nptxt(epoch_loss_value, precision, os.path.join(args.r, execution_time+'_training_loss.txt'))
          epoch_loss_value = 0
          batch_num = 0
          print("Current Epoch CI is%1.4f"%(epoch_CI/num_steps_per_train_epoch))
#      epoch_loss_value = epoch_loss_value + loss_value*args.bs  			                
    
#      if step % int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/args.bs) == 0:              			         
#          if epoch == args.me:
#              shutil.rmtree(args.d, ignore_errors=True)
#              break
          epoch = epoch + 1
          epoch_CI = 0
#      if step % 1 == 0:
#        num_examples_per_step = args.bs * FLAGS.num_gpus
#        examples_per_sec = num_examples_per_step / duration
#        sec_per_batch = duration / FLAGS.num_gpus
#        batch_num = batch_num + 1
#        print('\nTraining epoch: %s'%(epoch+1))
#        print('Batch number: %s'%batch_num)
#        format_str = ('%s: step %d, Train Log_Likelihood = %.4f (%.1f HPFs/sec; %.3f '
#                      'sec/batch)')
#        print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))         
      
#      if (step+1) % (model_params.saving_freq*num_steps_per_train_epoch) == 0 or (step + 1) == max_steps:
      if step % (model_params.saving_freq*num_steps_per_train_epoch) == 0 or step == max_steps:
        summary_str = sess.run(summary_op)
        summary_writer.add_summary(summary_str, step)
    
      # Save the model checkpoint periodically.
#      if (step+1) % (model_params.saving_freq*num_steps_per_train_epoch) == 0 or (step + 1) == max_steps:
      if step % (model_params.saving_freq*num_steps_per_train_epoch) == 0 or step == max_steps:
	if epoch > (args.me-args.nm):
	       	checkpoint_path = os.path.join(args.t, 'model.ckpt')
	       	saver.save(sess, checkpoint_path, global_step=step)
            

def main(argv=None):
  tf.gfile.MakeDirs(args.r)
  tf.gfile.MakeDirs(args.t)
  train()


if __name__ == '__main__':
  tf.app.run()
