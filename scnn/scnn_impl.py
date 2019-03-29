'''
file implements scnn image only option
'''
from __future__ import absolute_import, division, print_function
import os
import pandas as pd
import scnn.model_params as model_params
import scnn.model_tools as model_tools
import re
import tensorflow as tf
TRAIN_MODULITY = 'image_only'
features_path = os.path.join(os.getcwd(),'scnn','inputs','all_dataset.csv')
# read the all_dataset.csv file as a pandas data frame:
features = pd.read_csv(features_path, header=0)
# Get the 'TCGA ID' of the patients from data fram:
TCGA_IDs = features.iloc[:, 0]
# get the indexes, censored, Survival months:
all_features_org = features.iloc[:, 1:4]
# TCGA ID, indexes, censored, Survival months
all_features_org = pd.concat([TCGA_IDs, all_features_org], axis=1)

# find the index of the rows with NaN values in all_features data frame
nan_indexes = pd.isnull(all_features_org).any(1).nonzero()[0]
# remove rows with NaN values from all_features data frame:
all_features = all_features_org.drop(all_features_org.index[nan_indexes])
available_patients = all_features['TCGA ID'].tolist() # List of all the patients that have "not_NaN" features
number_of_labels = len(all_features.columns)-1 # number of labels in each binary file naming format: |Idh|Codeletion|Survival|censored|index|image|
image_train_path = os.path.join(os.getcwd(),'images','train')
data = model_tools.data_generator(image_train_path, available_patients) # A dictionary containing the list of images in Train set (data['train']) and Test set (data['test'])
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = len(data)
#create bin file location string
bin_file_base_path = os.path.join(os.getcwd(),'scnn','tmp')
bin_file_base_path += '\\'
trainset_base_name = bin_file_base_path + TRAIN_MODULITY + '_idx-censored-survival-image-GBMLGG_20x_' + str(
    model_params.org_image['height']) + 'x' + str(model_params.org_image['width']) + 'x' + str(
    model_params.org_image['channels']) + '_float16_train_'
train_chunk_name = trainset_base_name + '%d.bin'


