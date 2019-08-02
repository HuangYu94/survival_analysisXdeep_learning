import os
import tensorflow as tf
import math
import numpy as np
import pandas as pd

def get_imgID(img_path_name):
    img_id_start = img_path_name.find('TCGA')
    return img_path_name[img_id_start:img_id_start+12]
IMAGE_SIZE = 96

def eval_CI(Risk, T, C):
    """
    get performance evaluation.
    """
    def c_index_score_orderable(Risk, T, C):
        """
        Calculate concordance index to evaluate model prediction.
        C-index calulates the fraction of all pairs of subjects whose predicted
        survival times are correctly ordered among all subjects that can actually be ordered, i.e.
        both of them are uncensored or the uncensored time of one is smaller than
        the censored survival time of the other.


        Parameters
        ----------
        Risk: numpy.ndarray
           m sized array of predicted risk (do not confuse with predicted survival time!)
        T: numpy.ndarray
           m sized vector of time of death or last follow up
        C: numpy.ndarray
           m sized vector of censored status (do not confuse with observed status!)

        Returns
        -------
        A value between 0 and 1 indicating concordance index.
        """
        # count orderable pairs
        Orderable = 0.0
        Score = 0.0
        for i in range(len(T)):
            for j in range(i + 1, len(T)):
                if (C[i] == 0 and C[j] == 0):
                    Orderable = Orderable + 1
                    if (T[i] > T[j]):
                        if (Risk[j] > Risk[i]):
                            Score = Score + 1
                    elif (T[j] > T[i]):
                        if (Risk[i] > Risk[j]):
                            Score = Score + 1
                    else:
                        if (Risk[i] == Risk[j]):
                            Score = Score + 1
                elif (C[i] == 1 and C[j] == 0):
                    if (T[i] >= T[j]):
                        Orderable = Orderable + 1
                        if (T[i] > T[j]):
                            if (Risk[j] > Risk[i]):
                                Score = Score + 1
                elif (C[j] == 1 and C[i] == 0):
                    if (T[j] >= T[i]):
                        Orderable = Orderable + 1
                        if (T[j] > T[i]):
                            if (Risk[i] > Risk[j]):
                                Score = Score + 1

        return Score, Orderable


    def c_index(score, orderable):
        '''This function gets the score and orderable values and returns the
        c_index value.'''
        return score / orderable

    Score, Orderable = c_index_score_orderable(Risk, T, C)
    return c_index(Score, Orderable)

def log_sigmoid_loss(logits, labels, batch_size):
  """
  new loss directly optimize the differentiable log_sigmoid lower bound of CI
  :param logits: Logits from inference()
  :param labels: Labels from distorted_inputs or inputs(). 1-D tensor
            of shape [batch_size]
  :return: log_sigmoid_loss
  """
  idx_left = []
  idx_right = []
  for i in range(0, batch_size):
    for j in range(0, batch_size):
      if not i==j:
        idx_left.append(j)
        idx_right.append(i)

  risk_left = tf.gather(logits, idx_left)
  risk_right = tf.gather(logits, idx_right)
  mask = tf.cast(tf.gather(labels['survival'], idx_left) <= tf.gather(labels['survival'], idx_right),
                 dtype=tf.float32) * tf.cast(tf.gather(labels['observed'], idx_left), dtype=tf.float32)
  edge_sum = tf.reduce_sum(mask)
  loss = -(edge_sum + tf.reduce_sum(tf.log(tf.sigmoid(risk_left - risk_right))* mask))/math.log(2)/edge_sum
  tf.add_to_collection('losses', loss)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

def exponential_loss(logits, labels, batch_size):
  """
  new loss directly optimize the differentiable exponential lower bound of CI
  :param logits:
  :param labels:
  :param batch_size:
  :return:
  """
  idx_left = []
  idx_right = []
  for i in range(0, batch_size):
    for j in range(0, batch_size):
      if not i == j:
        idx_left.append(j)
        idx_right.append(i)

  risk_left = tf.gather(logits, idx_left)
  risk_right = tf.gather(logits, idx_right)
  mask = tf.cast(tf.gather(labels['survival'], idx_left) <= tf.gather(labels['survival'], idx_right),
                 dtype=tf.float32) * tf.cast(tf.gather(labels['observed'], idx_left), dtype=tf.float32)
  edge_sum = tf.reduce_sum(mask)
  loss = - (edge_sum - tf.reduce_sum(tf.exp(-(risk_left - risk_right)) * mask)) / edge_sum
  tf.add_to_collection('losses', loss)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')

all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_train_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'train', 'train_class')
all_df = pd.read_csv(all_data_csv_path)
num_cases, _ = all_df.shape

train_image_list = os.listdir(all_train_images_path)
batch_size = len(train_image_list)
some_train_img_list = []
train_ID_list = []
for img_str in train_image_list:
    some_train_img_list.append(os.path.join(all_train_images_path, img_str))
    train_ID_list.append(get_imgID(img_str))


all_df.set_index(keys='TCGA ID', inplace=True)
survival_val = []
censored_val = []
img_idx_list = [i for i in range(0, len(train_ID_list))]
for patient_id in train_ID_list:
    survival_val.append(int(all_df.loc[patient_id]['Survival months']))
    censored_val.append(int(all_df.loc[patient_id]['censored']))


risks = tf.Variable(tf.zeros(shape=(len(survival_val),)))
survival_ph = tf.placeholder(dtype=tf.int32, shape=(len(survival_val),))
censored_ph = tf.placeholder(dtype=tf.int32, shape=(len(censored_val),))
labels = {}
labels['survival'] = survival_ph
labels['censored'] = censored_ph
labels['observed'] = 1 - censored_ph

all_losses = exponential_loss(risks, labels, batch_size)
optimizer = tf.train.AdagradOptimizer(0.001)
train_var = tf.trainable_variables()
gradients = tf.gradients(all_losses, train_var)
train_op = optimizer.apply_gradients(zip(gradients, train_var))
num_epochs = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    prev_risks = np.zeros(shape=(len(survival_val),))
    for step_e in range(0, num_epochs):
        _, logits, loss_val, survival_value, censored_value = sess.run([train_op, risks, all_losses, labels['survival'],
                                                                        labels['censored']],
                                                                       feed_dict={survival_ph: np.asarray(survival_val),
                                                                                  censored_ph: np.asarray(censored_val)})
        new_risks = logits
        print('difference %1.4f'%(np.sum(np.abs(new_risks - prev_risks))))
        prev_risks = new_risks
        epoch_CI = eval_CI(logits, survival_val, censored_val)
        print("loss_value is %1.4f, CI is %1.4f"%(loss_val, epoch_CI))
    # np.savetxt('estimated_risk.txt', logits)
