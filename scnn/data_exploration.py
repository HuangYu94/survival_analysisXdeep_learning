import os
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input
import scnn.model_utils as utils
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

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

def SaveMetaData(all_data_df, img_lst, expriment, meta_path = "MetaData"):
    """
    Function to save feature meta data to meta data path. Meta data mainly contains survival information such as
    censored, survival months, patient ID.
    :param all_data_df: dataframe contains all information
    :param img_lst: image path list
    :param meta_path: meta data folder name
    :return: None
    """
    corresponding_id_lst = []
    censored_lst = []
    survival_months_lst = []
    for img_path in img_lst:
        img_id = utils.get_imgID(img_path)
        corresponding_id_lst.append(img_id)
        censored = all_data_df[all_data_df['TCGA ID'] == img_id]['censored'].values[0]
        censored_lst.append(censored)
        survival_months = all_data_df[all_data_df['TCGA ID'] == img_id]['Survival months'].values[0]
        survival_months_lst.append(survival_months)

    meta_path = os.path.join(os.path.dirname(__file__), meta_path)
    if not os.path.exists(meta_path):
        os.mkdir(meta_path)
    id_file = os.path.join(meta_path, expriment + "_patient_id.txt")
    censored_file = os.path.join(meta_path, expriment + '_censored.txt')
    survival_file = os.path.join(meta_path, expriment + '_survival.txt')
    np.savetxt(id_file, corresponding_id_lst, fmt='%s')
    np.savetxt(censored_file, np.asarray(censored_lst, dtype=np.int32))
    np.savetxt(survival_file, np.asarray(survival_months_lst, dtype=np.int32))


all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_train_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'train')
all_test_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test')
all_df = pd.read_csv(all_data_csv_path)
num_cases, _ = all_df.shape
train_image_list = os.listdir(all_train_images_path)
test_image_list = os.listdir(all_test_images_path)
some_train_img_list = []
some_test_image_list = []
for img_str in train_image_list:
    some_train_img_list.append(os.path.join(all_train_images_path, img_str))
for img_str in test_image_list:
    some_test_image_list.append(os.path.join(all_test_images_path, img_str))

model_dict = {}
model_dict['model name'] = "ResNet50"
model_dict['model'] = ResNet50(include_top=False, weights='imagenet')
model_dict['preprocess input'] = preprocess_input
# ExtractFeatures(model_dict, some_train_img_list, (224, 224, -1), "train")
# ExtractFeatures(model_dict, some_test_image_list, (224, 224, -1), "test")
# SaveMetaData(all_df, some_train_img_list, "train")
# SaveMetaData(all_df, some_test_image_list, "test")
model_dict.clear()

from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input

model_dict['model name'] = "Xception"
model_dict['model'] = Xception(include_top=False, weights='imagenet')
model_dict['preprocess input'] = preprocess_input
# ExtractFeatures(model_dict, some_train_img_list, (299, 299, -1), "train")
# ExtractFeatures(model_dict, some_test_image_list, (299, 299, -1), "test")
model_dict.clear()

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

model_dict['model name'] = "InceptionV3"
model_dict['model'] = InceptionV3(include_top=False, weights='imagenet')
model_dict['preprocess input'] = preprocess_input
# ExtractFeatures(model_dict, some_train_img_list, (299, 299, -1), "train")
# ExtractFeatures(model_dict, some_test_image_list, (299, 299, -1), "test")
model_dict.clear()

from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

model_dict['model name'] = "VGG16"
model_dict['model'] = VGG16(include_top=False, weights='imagenet')
model_dict['preprocess input'] = preprocess_input
# ExtractFeatures(model_dict, some_train_img_list, (224, 224, -1), "train")
# ExtractFeatures(model_dict, some_test_image_list, (224, 224, -1), "test")
model_dict.clear()

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

model_dict['model name'] = "InceptionResNetV2"
model_dict['model'] = InceptionResNetV2(include_top=False, weights='imagenet')
model_dict['preprocess input'] = preprocess_input
# ExtractFeatures(model_dict, some_train_img_list, (299, 299, -1), "train")
# ExtractFeatures(model_dict, some_test_image_list, (299, 299, -1), "test")
model_dict.clear()

from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import preprocess_input

model_dict['model name'] = "NASNetLarge"
model_dict['model'] = NASNetLarge(input_shape=(331, 331, 3), include_top=False, weights='imagenet')
model_dict['preprocess input'] = preprocess_input
ExtractFeatures(model_dict, some_train_img_list, (331, 331, -1), 'train')
ExtractFeatures(model_dict, some_test_image_list, (331, 331, -1), 'test')
model_dict.clear()

# corresponding_id_lst = []
# censored_lst = []
# survival_months_lst = []
# for idx, img_path in enumerate(some_train_img_list):
#     img = image.load_img(img_path, target_size=(224, 244, -1))
#     img_id = utils.get_imgID(img_path)
#     corresponding_id_lst.append(img_id)
#     img = np.expand_dims(image.img_to_array(img)[:, :, 0:3], axis=0)
#     x = preprocess_input(img)
#     censored = all_df[all_df['TCGA ID'] == img_id]['censored'].as_matrix()[0]
#     censored_lst.append(censored)
#     survival_months = all_df[all_df['TCGA ID'] == img_id]['Survival months'].as_matrix()[0]
#     survival_months_lst.append(survival_months)
#     if idx % 10 == 0 and idx > 1:
#         print("current process is %1.4f %%"%(idx/len(some_train_img_list) * 100))
#         print("shape of features is:")
#         print(features.shape)
#     if idx == 0:
#         feature = model.predict(x)
#         features = feature.reshape((1, -1))
#     else:
#         feature = model.predict(x)
#         features = np.concatenate((features, feature.reshape((1, -1))))
#
# np.savetxt('extracted.txt', features)
# np.savetxt('patient_id.txt', np.asarray(corresponding_id_lst), fmt='%s')
# np.savetxt('survival months.txt', np.asarray(survival_months_lst))
# np.savetxt('cencored.txt', np.asarray(censored_lst))

# features_centered = features - np.mean(features, axis=1).reshape((-1, 1))
# pca = PCA(n_components=50)
# features_low = pca.fit_transform(features)
#
# features_emb = TSNE(n_components=2).fit_transform(features_low)
# censored_lst = np.asarray(censored_lst, dtype=np.int).flatten()
# features_emb_censored = features_emb[censored_lst == 1]
# features_emb_notcensored = features_emb[censored_lst == 0]
# plt.scatter(features_emb_censored[:, 0], features_emb_censored[:, 1], marker='o',edgecolors='b')
# plt.scatter(features_emb_notcensored[:, 0], features_emb_notcensored[:, 1], marker='o', edgecolors='r')
# plt.show()



