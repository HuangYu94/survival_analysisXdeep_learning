import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image


def get_imgID(img_path_name):
    img_id_start = img_path_name.find('TCGA')
    return img_path_name[img_id_start:img_id_start+12]
IMAGE_SIZE = 96

def random_crop_batch(image_batch, random_crop_size = (IMAGE_SIZE, IMAGE_SIZE)):
    """
    randomly crop image batch into (IMAGE_SIZE, IMAGE_SIZE)
    :param image_batch: input image batch
    :param random_crop_size: 
    :return: cropped_batch: randomly cropped image batch with size (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3)
    :returns: pos_batch: position to crop (y, x) is the bot-left point; Shape: (batch_size, 2)
    """
    batch_size, img_hei, img_wid, _ = image_batch.shape
    dy, dx = random_crop_size

    cropped_batch = np.zeros(shape=(batch_size, IMAGE_SIZE, IMAGE_SIZE, 3))
    pos_batch = np.zeros(shape=(batch_size, 2), dtype=np.int32)
    for batch_idx in range(0, batch_size):
        x = np.random.randint(0, img_wid - dx + 1)
        y = np.random.randint(0, img_hei - dy + 1)
        cropped_img = np.asarray(image_batch[batch_idx, y:(y + dy), x:(x + dx), :], dtype=np.float32)
        cropped_img = cropped_img / (cropped_img.max() - cropped_img.min())
        cropped_batch[batch_idx, :, :, :] = cropped_img
        pos_batch[batch_idx, 0], pos_batch[batch_idx, 1] = y, x

    return cropped_batch, pos_batch

def ExtractFeatures(model_dict, img_path_lst, target_size, experiment, feature_path = "features"):
    """
    given a pretrained deep learning model, this function perform a forward pass of the all histology images
    and extract the feature from last layer and finally save it as txt file.
    :param model_dict: dictionary contains "model name", "model" and "preprocess_input" function(if there is one,
    can be "None")
    :param img_path_lst: list of path_to_images
    :param target_size: a tuple load image and reshape to legal input size
    :param experiment: "train" or "test"
    :param featuer_path: folder to store features
    :return: None
    """
    model = model_dict['model']
    model_name = model_dict['model name']
    preprocess_input_fn = model_dict['preprocess input']
    print("starting to extract feature from pretrained %s this will take a while..."%(model_name))
    for idx, img_path in enumerate(img_path_lst):
        img = image.load_img(img_path, target_size=target_size)
        img = np.expand_dims(image.img_to_array(img)[:, :, 0:3], axis=0)
        if preprocess_input_fn == None:
            x = img
        else:
            x = preprocess_input_fn(img)
        if idx % 10 == 0 and idx > 1:
            print("current process is %1.4f %%" % (idx / len(img_path_lst) * 100))
            print("shape of features is:")
            print(features.shape)
        if idx == 0:
            feature = model.predict(x)
            features = feature.reshape((1, -1))
        else:
            feature = model.predict(x)
            features = np.concatenate((features, feature.reshape((1, -1))))
    outfile_name = experiment + '_' + model_name + '_features.txt'
    feature_path = os.path.join(os.path.dirname(__file__), feature_path)
    if not os.path.exists(feature_path):
        os.mkdir(feature_path)
    outfile_name = os.path.join(feature_path, outfile_name)
    np.savetxt(outfile_name, features)



camelyon_model = load_model('CNN_keras_train.h5')
camelyon_model.summary()
layer_wanted = 'activation_6'
camelyon_model = Model(inputs=camelyon_model.input,
                       outputs=camelyon_model.get_layer(layer_wanted).output)


all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_train_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'train')
all_test_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test')
all_df = pd.read_csv(all_data_csv_path)
num_cases, _ = all_df.shape

model_dict = {}
model_dict['model name'] = "camelyon CNN pretrained"
model_dict['model'] = camelyon_model

Original_Image_Size = (1024, 1024)
batch_size = 32


datagen = image.ImageDataGenerator(preprocessing_function=lambda x: (x - x.mean()) / x.std() if x.std() > 0 else x,
                                   horizontal_flip=True,
                                   vertical_flip=True)

train_gen = datagen.flow_from_directory('..\\images\\train',
                                        target_size=Original_Image_Size,
                                        shuffle=True,
                                        batch_size=batch_size)
test_gen = datagen.flow_from_directory('..\\images\\test',
                                       target_size=Original_Image_Size,
                                       shuffle=False,
                                       batch_size=batch_size)



batches_per_epoch = train_gen.samples // train_gen.batch_size
num_batches_test = test_gen.samples // test_gen.batch_size
num_random_crop = 20
feature_dim = 256
all_df.set_index(keys='TCGA ID', inplace=True)
feature_dict = {}
feature_names = ['feature_train_cropped_' + str(i) for i in range(0, num_random_crop)]
for feature_name in feature_names:
    feature_dict[feature_name] = np.zeros(shape=(batches_per_epoch * batch_size, feature_dim))

for i in range(batches_per_epoch):
    batch_data, _ = train_gen.next()
    current_index = ((train_gen.batch_index-1) * train_gen.batch_size)
    if current_index < 0:
        if train_gen.samples % train_gen.batch_size > 0:
            current_index = max(0, train_gen.samples - train_gen.samples % train_gen.batch_size)
        else:
            current_index = max(0, train_gen.samples - train_gen.batch_size)
    index_array = train_gen.index_array[current_index:current_index + train_gen.batch_size].tolist()
    img_ID = list(map(get_imgID, [train_gen.filenames[idx] for idx in index_array]))
    if i%10 == 0:
        print("current training set process progress is %1.4f %%" % (i/batches_per_epoch * 100))
    for feature_name in feature_names:
        cropped_batch, pos_batch = random_crop_batch(batch_data)
        train_features = camelyon_model.predict(cropped_batch, batch_size=batch_size)
        feature_dict[feature_name][i*batch_size: (i+1)*batch_size, :] = train_features

    survival_batch, censored = all_df.loc[img_ID]['Survival months'].values, \
                               all_df.loc[img_ID]['censored'].values
    if i == 0:
        survival_data = survival_batch
        censored_data = censored
    else:
        survival_data = np.concatenate((survival_data, survival_batch), axis=0)
        censored_data = np.concatenate((censored_data, censored), axis=0)

# save train features
if not os.path.exists('.\\features'):
    os.mkdir('.\\features')

for feature_name in feature_names:
    np.savetxt(os.path.join('features', feature_name+'.txt'), feature_dict[feature_name])
np.savetxt(os.path.join('features', 'train_survival.txt'), survival_data)
np.savetxt(os.path.join('features', 'train_censor.txt'), censored_data)

feature_names = ['feature_test_cropped_' + str(i) for i in range(0, num_random_crop)]
for feature_name in feature_names:
    feature_dict[feature_name] = np.zeros(shape=(num_batches_test * batch_size, feature_dim))

for i in range(0, num_batches_test):
    batch_data, _ = test_gen.next()
    current_index = ((test_gen.batch_index-1) * test_gen.batch_size)
    if current_index < 0:
        if test_gen.samples % test_gen.batch_size > 0:
            current_index = max(0, test_gen.samples - test_gen.samples % test_gen.batch_size)
        else:
            current_index = max(0, test_gen.samples - test_gen.batch_size)
    index_array = test_gen.index_array[current_index:current_index + test_gen.batch_size].tolist()
    img_ID = list(map(get_imgID, [test_gen.filenames[idx] for idx in index_array]))
    if i%10 == 0:
        print("current testing set process progress is %1.4f %%" % (i/batches_per_epoch * 100))
    for feature_name in feature_names:
        cropped_batch, pos_batch = random_crop_batch(batch_data)
        train_features = camelyon_model.predict(cropped_batch, batch_size=batch_size)
        feature_dict[feature_name][i*batch_size: (i+1)*batch_size, :] = train_features

    survival_batch, censored = all_df.loc[img_ID]['Survival months'].values, \
                               all_df.loc[img_ID]['censored'].values

    if i == 0:
        survival_data = survival_batch
        censored_data = censored
    else:
        survival_data = np.concatenate((survival_data, survival_batch), axis=0)
        censored_data = np.concatenate((censored_data, censored), axis=0)

for feature_name in feature_names:
    np.savetxt(os.path.join('features', feature_name + '.txt'), feature_dict[feature_name])
np.savetxt(os.path.join('features', 'test_survival.txt'), survival_data)
np.savetxt(os.path.join('features', 'test_censor.txt'), censored_data)






    # testing











