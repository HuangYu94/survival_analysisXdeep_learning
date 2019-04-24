import os
import numpy as np
import pandas as pd
import tensorflow as tf

def LoadFeatures(feature_dir, model_name):
    """
    function to load preprocessed deep features from specified model
    :param feature_dir: directory name contains preprocessed feature files
    :param model_name: name of the model
    :return: train_features: features for training
    :return: test_features: features for testing
    """
    train_fname = os.path.join(feature_dir, 'train_' + model_name + '_features.txt')
    train_features = np.loadtxt(train_fname)
    test_fname = os.path.join(feature_dir, 'test_' + model_name + '_features.txt')
    test_features = np.loadtxt(test_fname)
    return train_features, test_features

def LoadMeta(meta_dir):
    """
    load meta data from meta data directory
    :param meta_dir: directory storing meta data files
    :return: train_meta_dict: train dictionary contains meta data ['patient_id', 'censored', 'survival months']
    :return: test_meta_dict: test dictionary contains meta data ['patient_id', 'censored', 'survival months']
    """
    train_meta_dict = {}
    train_meta_dict['patient_id'] = np.genfromtxt(os.path.join(meta_dir, 'train_patient_id.txt'))
    train_meta_dict['survival months'] = np.loadtxt(os.path.join(meta_dir, 'train_survival.txt'))
    train_meta_dict['censored'] = np.loadtxt(os.path.join(meta_dir, 'train_censored.txt'))
    test_meta_dict = {}
    test_meta_dict['patient_id'] = np.genfromtxt(os.path.join(meta_dir, 'test_patient_id.txt'))
    test_meta_dict['survival months'] = np.loadtxt(os.path.join(meta_dir, 'test_survival.txt'))
    test_meta_dict['censored'] = np.loadtxt(os.path.join(meta_dir, 'test_censored.txt'))
    return train_meta_dict, test_meta_dict


def calc_at_risk(X, Y, batch_size):
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

    values, order = tf.nn.top_k(Y['survival'], batch_size, sorted=True)
    values = tf.reverse(values, axis=[-1])
    order = tf.reverse(order, axis=[-1])
    sorted_survivals = values
    Y['at_risk'] = tf.nn.top_k(sorted_survivals, batch_size, sorted=False).indices
    Y['survival'] = sorted_survivals
    Y['observed'] = tf.gather(Y['observed'], order)
    X = tf.gather(X, order)
    # Y['idx'] = tf.gather(Y['idx'], order)
    return X, Y


def loss(logits, labels):
    """Add L2Loss to all the trainable variables.
    Add summary for "Loss" and "Loss/avg".
    Args:
      logits: Logits from inference().
      labels: Labels from distorted_inputs or inputs(). 1-D tensor
              of shape [batch_size]
    Returns:
      Loss tensor of type float.
    """
    factorized_logits = logits - tf.reduce_max(logits)  # subtract maximum to facilitate computation
    exp = tf.exp(factorized_logits)
    partial_sum = tf.cumsum(exp, reverse=True)  # get the reversed partial cumulative sum
    log_at_risk = tf.log(tf.gather(partial_sum, tf.reshape(labels['at_risk'], [-1]))) + tf.reduce_max(
        logits)  # add maximum back
    diff = tf.subtract(logits, log_at_risk)
    times = tf.reshape(diff, [-1]) * tf.cast(labels['observed'], tf.float32)
    cost = - (tf.reduce_sum(times))
    tf.add_to_collection('losses', cost)

    return tf.add_n(tf.get_collection('losses'), name='total_loss')

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def initializer(stddev):
    initializer = {
        'xavier': tf.contrib.layers.xavier_initializer(),
        'truncated_normal': tf.truncated_normal_initializer(stddev=stddev),
        'variance_scaling': tf.contrib.layers.variance_scaling_initializer()
    }

    return initializer

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  var = _variable_on_cpu(name, shape, initializer(stddev)['variance_scaling'])
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

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

class DenseModel(object):
    def __init__(self, input_placeholder, layer_specs):
        self.num_layers = len(layer_specs)
        self.all_params = {}
        self.input_placeholder = input_placeholder
        for layer_idx, layer_unit_num in enumerate(layer_specs):
            with tf.variable_scope('layer_' + str(layer_idx)):
                if layer_idx == 0:
                    weights = _variable_with_weight_decay('weights', (int(input_placeholder.shape[1]), layer_unit_num),
                                                          stddev=0.001, wd=1e-6)
                    last_out = input_placeholder
                else:
                    weights = _variable_with_weight_decay('weights', (layer_specs[layer_idx-1], layer_unit_num),
                                                          stddev=0.001, wd=1e-6)
                    last_out = out
                self.all_params['layer' + str(layer_idx) + '_weights'] = weights
                bias = _variable_on_cpu('bias', [layer_unit_num], tf.constant_initializer(0.0))
                self.all_params['layer' + str(layer_idx) + '_bias'] = bias
                out = tf.nn.bias_add(tf.matmul(last_out, weights), bias)

        self.out = out



batch_size = 64
feature_dir = os.path.join(os.path.dirname(__file__), 'features')
meta_dir = os.path.join(os.path.dirname(__file__), 'MetaData')

train_features, test_features = LoadFeatures(feature_dir, 'ResNet50')
num_cases_train, feature_dim = train_features.shape
print(train_features.shape)
print(test_features.shape)
train_meta_dict, test_meta_dict = LoadMeta(meta_dir)
print(train_meta_dict['survival months'].shape)
print(train_meta_dict['censored'].shape)
print(test_meta_dict['survival months'].shape)
num_cases_test, _ = test_features.shape


feature_ph = tf.placeholder(dtype=tf.float32, shape=[batch_size, feature_dim])
censored_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size])
survival_ph = tf.placeholder(dtype=tf.int32, shape=[batch_size])

label = {}
label['survival'] = survival_ph
label['censored'] = censored_ph
label['observed'] = 1 - censored_ph
X, Y = calc_at_risk(feature_ph, label, batch_size)
dense_model = DenseModel(X, [1000, 256, 1])
all_losses = loss(dense_model.out, Y)

optimizer = tf.train.AdagradOptimizer(0.0001)
train_var = tf.trainable_variables()
gradients = tf.gradients(all_losses, train_var)
train_op = optimizer.apply_gradients(zip(gradients, train_var))

num_epoch = 1000
num_batches = num_cases_train // batch_size
num_batches_test = num_cases_test // batch_size
train_case_idx_lst = [i for i in range(0, num_cases_train)]
test_case_idx_lst = [i for i in range(0, num_cases_test)]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch_step in range(0, num_epoch):
        np.random.shuffle(train_case_idx_lst)
        epoch_loss = 0
        for batch_i in range(0, num_batches-1):
            grab_idx = train_case_idx_lst[batch_i*batch_size: (batch_i+1)*batch_size]
            for _ in range(0, 100):
                _, batch_loss = sess.run([train_op, all_losses],
                                         feed_dict={feature_ph: train_features[grab_idx, :],
                                                    censored_ph: train_meta_dict['censored'][grab_idx],
                                                    survival_ph: train_meta_dict['survival months'][grab_idx]})
            epoch_loss += batch_loss/batch_size
        last_batch_idx = train_case_idx_lst[(num_batches-1)*batch_size: num_batches*batch_size]
        batch_loss_val, risk_val = sess.run([all_losses, dense_model.out],
                                            feed_dict={feature_ph: train_features[last_batch_idx, :],
                                                       censored_ph: train_meta_dict['censored'][last_batch_idx],
                                                       survival_ph: train_meta_dict['survival months'][last_batch_idx]})
        batch_loss_val = batch_loss_val/batch_size
        epoch_loss = epoch_loss/num_batches
        val_CI = eval_CI(risk_val, train_meta_dict['survival months'][last_batch_idx],
                         train_meta_dict['censored'][last_batch_idx])
        print("epoch %d finish with epoch train loss %1.4f validation loss %1.4f validation CI %1.4f" %
              (epoch_step, epoch_loss, batch_loss_val, val_CI))
        # doing testing
        test_CI = 0
        for batch_i in range(0, num_batches_test):
            grab_idx = test_case_idx_lst[batch_i*batch_size: (batch_i+1)*batch_size]
            risk_test = sess.run(dense_model.out, feed_dict={feature_ph: test_features[grab_idx, :],
                                                             censored_ph: test_meta_dict['censored'][grab_idx],
                                                             survival_ph: test_meta_dict['survival months'][grab_idx]})
            test_CI += eval_CI(risk_test, test_meta_dict['survival months'][grab_idx], test_meta_dict['censored'][grab_idx])
        test_CI = test_CI / num_batches_test
        print("test CI is %1.4f" % test_CI)



