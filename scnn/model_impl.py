import numpy as np
import tensorflow as tf
import time
import datetime

# define model parameters in the following:
num_output_units = 1
Keep_Prob = 0.95
Batch_Size = 32
Optimizer_Name = 'gd'
Num_GPUs = 1
Tower_Name = 'tower'
Moving_Average_Decay = 0.9999
Learing_Rate_Decay_Factor = 0.1
Learning_Rate = 0.001
Num_of_Model_Averaging = 1  # number of models to be averaged at test time
Num_of_Epochs_Per_Decay = 1500
Max_Steps = 100  # max number of batches
Max_Num_Epoch = 100  # max number of epoch to run

# define global parameters but are not model parameters
num_steps_per_train_epoch = 10
num_batches = 10
train_log_folder = 'train_log'
epoch_loss_fname = 'epoch_loss.txt'
test_log_folder = 'test_log'
train_ckpt_folder = 'checkpoints'

def initializer(stddev):
    initializer = {
        'xavier': tf.contrib.layers.xavier_initializer(),
        'truncated_normal': tf.truncated_normal_initializer(stddev=stddev),
        'variance_scaling': tf.contrib.layers.variance_scaling_initializer(),
        'he_normal': tf.keras.initializers.he_normal()
    }

    return initializer


def optimizer(lr):
    optimizer = {
        'gd': tf.train.GradientDescentOptimizer(lr),
        'adagrad': tf.train.AdagradOptimizer(lr, initial_accumulator_value=0.1),
        'adadelta': tf.train.AdadeltaOptimizer(lr),
        'adam': tf.train.AdamOptimizer(lr)
    }

    return optimizer

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
  # with tf.device('/device:GPU:0'):
    var = tf.get_variable(name, shape, initializer=initializer)
  return var


def _variable_with_weight_decay(name, shape, stddev, wd, initializer_w='variance_scaling'):
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
  var = _variable_on_cpu(name, shape, initializer(stddev)[initializer_w])
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var



def distorted_inputs(tfrecord_iterator, batch_size=64, crop_size=256):
    """
    Making disttorted input images for training, image in shape (batch_size,
    :param tfrecord_iterator:
    can be unpacked as survival_months, censored, patient_ID, histology_image = tfrecord_iterator.get_next()
    crop_size: parameter for random_crop(), image will be cropped into (crop_size, crop_size)
    :return: (survival_months, censored, patient_ID, histology_image_distorted)
    """

    survival_months, censored, patient_ID, histology_image = tfrecord_iterator.get_next()
    histology_image = tf.cast(histology_image, tf.float32)
    img_hei, img_wid, img_channel = histology_image.shape[1], histology_image.shape[2], histology_image.shape[3]

    height = crop_size
    width = crop_size

    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    histology_image_ditorted_lst = []
    for case_idx in range(0, batch_size):
        histology_image_tmp = histology_image[case_idx]
        # Randomly crop a [height, width] section of the image.
        histology_image_ditorted = tf.random_crop(histology_image_tmp, [height, width, img_channel])

        # Randomly flip the image horizontally.
        histology_image_ditorted = tf.image.random_flip_left_right(histology_image_ditorted)

        # Because these operations are not commutative, consider randomizing
        # the order their operation.
        histology_image_ditorted = tf.image.random_brightness(histology_image_ditorted,
                                                   max_delta=63)
        histology_image_ditorted = tf.image.random_contrast(histology_image_ditorted,
                                                 lower=0.2, upper=1.8)

        # Subtract off the mean and divide by the variance of the pixels in TensorFlow v0.12.
        histology_image_ditorted = tf.image.per_image_standardization(histology_image_ditorted)
        histology_image_ditorted_lst.append(histology_image_ditorted)

    histology_image_ditorted = tf.stack(histology_image_ditorted_lst)

    return survival_months, censored, patient_ID, histology_image_ditorted



def inference(images, keep_prob, batch_size, img_channel = 3):
    """Build the cnn model. Labels would be useful here
    Args:
      images: Images returned from distorted_inputs() or inputs().
      keep_prob: Dropout probability
    Returns:
      Logits.
    """
    # We instantiate all variables using tf.get_variable() instead of
    # tf.Variable() in order to share variables across multiple GPU training runs.
    # If we only ran this model on a single GPU, we could simplify this function
    # by replacing all instances of tf.get_variable() with tf.Variable().
    #
    # conv1
    with tf.variable_scope('conv1') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, img_channel, 64],
                                             stddev=1e-4,
                                             wd=0.0)  # Number of Channels = 1 because the images are grey scale
        conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv1)

    # norm1
    norm1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 64], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv2)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')

    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # conv3
    with tf.variable_scope('conv3') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 64, 128], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv3)

    # norm3
    norm3 = tf.nn.lrn(conv3, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm3')

    # conv4
    with tf.variable_scope('conv4') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 128], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [128], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv4 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv4)

    # norm4
    norm4 = tf.nn.lrn(conv4, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm4')

    # pool4
    pool4 = tf.nn.max_pool(norm4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

    # conv5
    with tf.variable_scope('conv5') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 128, 256], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(pool4, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv5)

    # norm5
    norm5 = tf.nn.lrn(conv5, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm5')

    # conv6
    with tf.variable_scope('conv6') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm5, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv6 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv6)

    # norm6
    norm6 = tf.nn.lrn(conv6, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm6')

    # conv7
    with tf.variable_scope('conv7') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm6, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv7 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv7)

    # norm7
    norm7 = tf.nn.lrn(conv7, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm7')

    # conv8
    with tf.variable_scope('conv8') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 256], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm7, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv8 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv8)

    # norm8
    norm8 = tf.nn.lrn(conv8, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm8')

    # pool8
    pool8 = tf.nn.max_pool(norm8, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool8')

    # conv9
    with tf.variable_scope('conv9') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 256, 512], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(pool8, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv9 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv9)

    # norm9
    norm9 = tf.nn.lrn(conv9, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm9')

    # conv10
    with tf.variable_scope('conv10') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm9, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv10 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv10)

    # norm10
    norm10 = tf.nn.lrn(conv10, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm10')

    # conv11
    with tf.variable_scope('conv11') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm10, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv11 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv11)

    # norm11
    norm11 = tf.nn.lrn(conv11, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm11')

    # conv12
    with tf.variable_scope('conv12') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm11, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv12 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv12)

    # norm12
    norm12 = tf.nn.lrn(conv12, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm12')

    # pool12
    pool12 = tf.nn.max_pool(norm12, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool12')

    # conv13
    with tf.variable_scope('conv13') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(pool12, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv13 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv13)

    # norm13
    norm13 = tf.nn.lrn(conv13, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm13')

    # conv14
    with tf.variable_scope('conv14') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm13, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv14 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv14)

    # norm14
    norm14 = tf.nn.lrn(conv14, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm14')

    # conv15
    with tf.variable_scope('conv15') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm14, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv15 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv15)

    # norm15
    norm15 = tf.nn.lrn(conv15, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm15')

    # conv16
    with tf.variable_scope('conv16') as scope:
        kernel = _variable_with_weight_decay('weights', shape=[3, 3, 512, 512], stddev=1e-4, wd=0.0)
        conv = tf.nn.conv2d(norm15, kernel, [1, 1, 1, 1], padding='SAME')
        biases = _variable_on_cpu('biases', [512], tf.constant_initializer(0.1))
        bias = tf.nn.bias_add(conv, biases)
        conv16 = tf.nn.relu(bias, name=scope.name)
        #_activation_summary(conv16)

    # norm16
    norm16 = tf.nn.lrn(conv16, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm16')

    # pool16
    pool16 = tf.nn.max_pool(norm16, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool16')

    # local17
    with tf.variable_scope('local17') as scope:
        # Move everything into depth so we can perform a single matrix multiply.
        reshape = tf.reshape(pool16, [batch_size, -1])
        # if experiment == 'image_genome':
        #     reshape_genomic_added = tf.concat_v2([reshape, labels['genomics']], 1)
        #     reshape = reshape_genomic_added
        dim = reshape.get_shape()[1].value
        weights = _variable_with_weight_decay('weights', shape=[dim, 1000], stddev=0.04, wd=0.0004)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0.1))
        local17 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)
        #_activation_summary(local17)

    # local18
    with tf.variable_scope('local18') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1000, 1000], stddev=0.04, wd=0.0004)
        biases = _variable_on_cpu('biases', [1000], tf.constant_initializer(0.1))
        local18 = tf.nn.relu(tf.matmul(local17, weights) + biases, name=scope.name)
        #_activation_summary(local18)

    # local19
    with tf.variable_scope('local19') as scope:
        weights = _variable_with_weight_decay('weights', shape=[1000, 256], stddev=0.04, wd=0.0004)
        biases = _variable_on_cpu('biases', [256], tf.constant_initializer(0.1))
        local19 = tf.nn.relu(tf.matmul(local18, weights) + biases, name=scope.name)
        local19_drop = tf.nn.dropout(local19, keep_prob)
        #_activation_summary(local19_drop)

    # softmax, i.e. softmax(WX + b)
    with tf.variable_scope('softmax_linear') as scope:
        weights = _variable_with_weight_decay('weights', [256, num_output_units], stddev=1 / 1000.0, wd=0.0)
        biases = _variable_on_cpu('biases', [num_output_units], tf.constant_initializer(0.0))
        softmax_linear = tf.add(tf.matmul(local19_drop, weights), biases, name=scope.name)
        #_activation_summary(softmax_linear)

    return softmax_linear


def calc_at_risk(histology_image, survival_months, censored, batch_size=32):
    """
    Helper function to ease the construction of loss function in Cox Partial likelihood
    :param histology_image: tf.Tensor in shape [batch_size, hei, wid, channel] contains histological image
    :param survival_months: tf.Tensor in shape [batch_size] contains number of months patients survived
    :param censored: tf.Tensor in shape [batch_size] contains censoring information
    :return:
    histology_image_ordered: ordered histological images ranked in ascending order of survival months in shape
    [batch_size, hei, wid, channel]
    labels: dictionary contains ordered index('at_risk'), ordered survival months('survival'), patients whose deaths
    observed('observed') all in shape [batch_size]
    """
    values, order = tf.nn.top_k(survival_months, batch_size, sorted=True)
    values = tf.reverse(values, axis=[0])
    order = tf.reverse(order, axis=[0])
    sorted_survivals = values
    labels = {}
    labels['at_risk'] = tf.nn.top_k(sorted_survivals, batch_size, sorted=False).indices
    labels['survival'] = sorted_survivals
    labels['observed'] = tf.gather(1 - censored, order)
    histology_image_ordered = tf.gather(histology_image, order)
    # if experiment == 'image_genome':
    #     Y['genomics'] = tf.gather(Y['genomics'], order)

    return histology_image_ordered, labels


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


def tower_loss(scope, tfrecord_iterator, batch_size=64):
    """
    function to compute total loss given data set
    :param scope: naming scope to get all losses
    :param tfrecord_iterator: tfrecord iterator to get all data and labels by calling .next()
    :return:
    """
    survival_months, censored, patient_ID, histology_image_ditorted = distorted_inputs(tfrecord_iterator, batch_size)
    images, labels = calc_at_risk(histology_image_ditorted, survival_months, censored, batch_size)
    # if args.m == 'image_genome':
    #     labels['genomics'] = tf.cast(labels['genomics'], tf.float32)

    logits = inference(images, Keep_Prob, Batch_Size)
    _ = loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    # for l in losses + [total_loss]:
    #     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    #     # session. This helps the clarity of presentation on tensorboard.
    #     loss_name = re.sub('%s_[0-9]*/' % model.TOWER_NAME, '', l.op.name)
    #     tf.summary.scalar(loss_name, l)

    # "labels" is a dictionary in which the keywords are among: 'idx' for
    # patients' indexes, 'survival', 'censored', 'observed', 'idh', 'codel',
    # 'copynum' for copy numbers, 'at_risk'
    return total_loss, logits, labels, images

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
        opt = optimizer(lr)[Optimizer_Name]

        # Calculate the gradients for each model tower.
        tower_grads = []
        for i in range(Num_GPUs):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (Tower_Name, i)) as scope:
                    # Calculate the loss for one tower of the cnn model. This function
                    # constructs the entire cnn model but shares the variables across
                    # all towers.
                    #          loss, logits, labels, images, observed, indexes, copy_numbers = tower_loss(scope)

                    # Reuse variables for the next tower.
                    # tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
                        loss, logits, labels, images = tower_loss(scope)

                        # Retain the summaries from the final tower.
                        # summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)

                        # Calculate the gradients for the batch of data on this cnn tower.
                        grads = opt.compute_gradients(loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

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
            Moving_Average_Decay, global_step)
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
        for step in range(1, Max_Steps + 1):
            # if epoch == args.me:
            #     shutil.rmtree(args.d, ignore_errors=True)
            #     break

            #      if step != 0:
            start_time = time.time()
            _, loss_value, logits_value, labels_value, images_value = sess.run([train_op, loss, logits, labels, images])
            duration = time.time() - start_time
            #      batch_num = batch_num + 1

            #      if step % 1 == 0:
            num_examples_per_step = Batch_Size * Num_GPUs
            examples_per_sec = num_examples_per_step / duration
            sec_per_batch = duration / Num_GPUs
            batch_num = batch_num + 1
            print('\nTraining epoch: %s' % (epoch + 1))
            print('Batch number: %s' % batch_num)
            format_str = ('%s: step %d, Train Log_Likelihood = %.4f (%.1f HPFs/sec; %.3f '
                          'sec/batch)')
            print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))

            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

            epoch_loss_value = epoch_loss_value + loss_value * Batch_Size
            # if step % num_steps_per_train_epoch == 0:
            #     epoch_loss_value = epoch_loss_value / (num_steps_per_train_epoch * Batch_Size)
            #     model_tools.nptxt(epoch_loss_value, precision,
            #                       os.path.join(args.r, execution_time + '_training_loss.txt'))
            #     epoch_loss_value = 0
            #     batch_num = 0
            #     #      epoch_loss_value = epoch_loss_value + loss_value*args.bs
            #
            #     #      if step % int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN/args.bs) == 0:
            #     #          if epoch == args.me:
            #     #              shutil.rmtree(args.d, ignore_errors=True)
            #     #              break
            #     epoch = epoch + 1

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
            # if step % (model_params.saving_freq * num_steps_per_train_epoch) == 0 or step == max_steps:
            #     summary_str = sess.run(summary_op)
            #     summary_writer.add_summary(summary_str, step)

            # Save the model checkpoint periodically.
            #      if (step+1) % (model_params.saving_freq*num_steps_per_train_epoch) == 0 or (step + 1) == max_steps:
            # if step % (model_params.saving_freq * num_steps_per_train_epoch) == 0 or step == max_steps:
            #     if epoch > (args.me - args.nm):
            #         checkpoint_path = os.path.join(args.t, 'model.ckpt')
            #         saver.save(sess, checkpoint_path, global_step=step)

