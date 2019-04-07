import tensorflow as tf


num_output_units = 1

def initializer(stddev):
    initializer = {
        'xavier': tf.contrib.layers.xavier_initializer(),
        'truncated_normal': tf.truncated_normal_initializer(stddev=stddev),
        'variance_scaling': tf.contrib.layers.variance_scaling_initializer(),
        'he_normal': tf.keras.initializers.he_normal()
    }

    return initializer

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



def distorted_inputs(tfrecord_iterator, crop_size=256):
    """
    Making disttorted input images for training, image in shape (batch_size,
    :param tfrecord_iterator:
    can be unpacked as survival_months, censored, patient_ID, histology_image = tfrecord_iterator.get_next()
    crop_size: parameter for random_crop(), image will be cropped into (crop_size, crop_size)
    :return: (survival_months, censored, patient_ID, histology_image_distorted)
    """

    survival_months, censored, patient_ID, histology_image = tfrecord_iterator.get_next()
    histology_image = tf.cast(histology_image, tf.float32)
    img_hei, img_wid, img_channel = histology_image.shape

    height = crop_size
    width = crop_size

    # Image processing for training the network. Note the many random
    # distortions applied to the image.
    histology_image_unstacked = tf.unstack(histology_image) # random operation can only be applied to 1 image
    histology_image_ditorted_lst = []
    for histology_image_tmp in histology_image_unstacked:
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



