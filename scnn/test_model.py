import os
import scnn.model_impl as model
import scnn.model_utils as utils
import time
from datetime import datetime
import numpy as np
import pandas as pd
import tensorflow as tf





def Evaluate(saved_model):
    with tf.Graph().as_default() as g:

        with tf.variable_scope(tf.get_variable_scope(), reuse=False) as scope:
            test_logits, test_labels = test_tower_loss(scope)

            # Restore the moving average version of the learned variables for eval.
        variable_averages = tf.train.ExponentialMovingAverage(
            model.Moving_Average_Decay)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

    results_df = eval_once(saved_model, saver, test_logits, test_labels)

    return results_df



num_models = model.Num_of_Model_Averaging
all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_test_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test')
all_df = pd.read_csv(all_data_csv_path)
test_image_list = os.listdir(all_test_images_path)
# convert test data into .tfrecords



model_results = {}
for saved_model in range(0, num_models):



