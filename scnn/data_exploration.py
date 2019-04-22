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

all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_train_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'train')
all_df = pd.read_csv(all_data_csv_path)
num_cases, _ = all_df.shape
train_image_list = os.listdir(all_train_images_path)
some_train_img_list = []
for img_str in train_image_list:
    some_train_img_list.append(os.path.join(all_train_images_path, img_str))


model = ResNet50(include_top=False, weights='imagenet')
corresponding_id_lst = []
censored_lst = []
survival_months_lst = []
for idx, img_path in enumerate(some_train_img_list):
    img = image.load_img(img_path, target_size=(224, 244, -1))
    img_id = utils.get_imgID(img_path)
    corresponding_id_lst.append(img_id)
    img = np.expand_dims(image.img_to_array(img)[:, :, 0:3], axis=0)
    x = preprocess_input(img)
    corresponding_id_lst.append(img_id)
    censored = all_df[all_df['TCGA ID'] == img_id]['censored'].as_matrix()[0]
    censored_lst.append(censored)
    survival_months = all_df[all_df['TCGA ID'] == img_id]['Survival months'].as_matrix()[0]
    if idx % 10 == 0 and idx > 1:
        print("current process is %1.4f %%"%(idx/len(some_train_img_list) * 100))
        print("shape of features is:")
        print(features.shape)
    if idx == 0:
        feature = model.predict(x)
        features = feature.reshape((1, -1))
    else:
        feature = model.predict(x)
        features = np.concatenate((features, feature.reshape((1, -1))))

np.savetxt('extracted.txt', features)

features_centered = features - np.mean(features, axis=1).reshape((-1,1))
pca = PCA(n_components=50)
features_low = pca.fit_transform(features)

features_emb = TSNE(n_components=2).fit_transform(features_low)
censored_lst = np.asarray(censored_lst, dtype=np.int).flatten()
features_emb_censored = features_emb[censored_lst == 1]
features_emb_notcensored = features_emb[censored_lst == 0]
plt.scatter(features_emb_censored[:, 0], features_emb_censored[:, 1], marker='o',edgecolors='b')
plt.scatter(features_emb_notcensored[:, 0], features_emb_notcensored[:, 1], marker='o', edgecolors='r')
plt.show()



