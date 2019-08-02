import os
import numpy as np
from scipy import ndimage as ndi
import pandas as pd
import tensorflow as tf
from PIL import Image
from skimage.color import rgba2rgb
from skimage.color import rgb2gray
from skimage.feature import greycomatrix
from skimage.feature import greycoprops
from skimage.feature import local_binary_pattern, hog
from skimage.filters import gabor_kernel
from skimage.util import img_as_float

import matplotlib.pyplot as plt

np.random.seed(42)
def get_imgID(img_path_name):
    img_id_start = img_path_name.find('TCGA')
    return img_path_name[img_id_start:img_id_start+12]
IMAGE_SIZE = 96

def im2uint8(im_float):
    """
    convert float image to uint8
    """
    return np.asarray(im_float * 256, dtype=np.uint8)


def random_crop_gray(image, random_crop_size = (IMAGE_SIZE, IMAGE_SIZE)):
    """
    randomly crop image batch into (IMAGE_SIZE, IMAGE_SIZE)
    :param image_batch: input image batch
    :param random_crop_size:
    :return: cropped_batch: randomly cropped image batch with size (batch_size, IMAGE_SIZE, IMAGE_SIZE, 3)
    :returns: pos_batch: position to crop (y, x) is the bot-left point; Shape: (batch_size, 2)
    """
    img_hei, img_wid = image.shape
    dy, dx = random_crop_size

    x = np.random.randint(0, img_wid - dx + 1)
    y = np.random.randint(0, img_hei - dy + 1)
    cropped_img = image[y:(y + dy), x:(x + dx)]
    return cropped_img, (y, x)

def evenlyspace(image):
    """
    hard code function to space image evenly into 8x8.
    bad performance 0.53
    """
    dy, dx = 128, 128
    patches = np.zeros(shape=(64, 128, 128))
    for i in range(0, 8):
        for j in range(0, 8):
            patches[i+j, :, :] = image[i*dy:(i+1)*dy, j*dx:(j+1)*dx]

    return patches


def get_image_feature_GLCM(img_dir, neib_specs=(1,), ang_specs=(0, np.pi/4, np.pi/2, np.pi),  crop_num = 16):
    """
    function to generate image GLCM feature
    """
    first_image = Image.open(img_dir)
    first_image_gray = rgb2gray(rgba2rgb(first_image))
    # plot_0 = plt.figure(0).add_subplot(111).imshow(first_image_gray, cmap='gray')
    # plt.show()
    # print(first_image_gray.shape)
    prop_specs = ['contrast', 'dissimilarity', 'homogeneity', 'ASM']

    feature_mat = np.zeros((len(prop_specs), crop_num, len(neib_specs), len(ang_specs)))
    for i, prop_spec in enumerate(prop_specs):
        for j in range(0, crop_num):
            cropped_image, _ = random_crop_gray(first_image_gray)
            greycomatrix_val = greycomatrix(im2uint8(cropped_image), neib_specs, ang_specs)
            feature_mat[i, j, :, :] = greycoprops(greycomatrix_val, prop=prop_spec)

    # print(greycomatrix_val_contrast.shape)
    # plot_1 = plt.figure(1).add_subplot(111).imshow(greycomatrix_val_contrast)
    # plt.show()
    return feature_mat.flatten()

def get_image_feature_LBP(img_dir, radius = 4, n_direction = 8, METHOD='uniform', num_crops = 4):
    """
    function to extraction local binary pattern given image
    :param img_dir: image directory
    :param radius: number of pixels in the radius of LBP
    :param n_direction: number of pixel radius direction
    :param METHOD: use uniform to avoid crash!
    :return: LBP histogram feature vector
    """
    image_gray = rgb2gray(rgba2rgb(Image.open(img_dir)))
    for i in range(0, num_crops):
        croped_gray, _ = random_crop_gray(image_gray, random_crop_size=(256, 256))
        lbp = local_binary_pattern(croped_gray, radius*n_direction, radius, METHOD)
        if i == 0:
            hist_ret, _ = np.histogram(lbp.ravel(), bins=radius*n_direction*2, density=True)
        else:
            hist, _ = np.histogram(lbp.ravel(), bins=radius*n_direction*2, density=True)
            hist_ret = np.concatenate((hist_ret, hist), axis=0)
    return hist_ret

def get_image_feature_HOG(img_dir, num_orientations=8, ppc=16, cpb=(2,2), num_crops = 4):
    """
    function to extract histogram of oriented gradient feature given image
    :param img_dir: image director
    :param num_orientations: number of orientations in computing HOG
    :param ppc: pixels per cell
    :param cpb: cells per block
    :param num_crops: how many patches to be cropped
    :return: hist_ret: HOG histogram feature vector
    """
    num_bins_each = 64
    image_gray = rgb2gray(rgba2rgb(Image.open(img_dir)))
    for i in range(0, num_crops):
        croped_gray, _ = random_crop_gray(image_gray, random_crop_size=(256, 256))
        img_hog, _ = hog(croped_gray, orientations=num_orientations, pixels_per_cell=(ppc, ppc), cells_per_block=cpb, block_norm='L2', visualise=True)
        if i == 0:
            hist_ret, _ = np.histogram(img_hog.ravel(), bins=num_bins_each, density=True)
        else:
            hist, _ = np.histogram(img_hog.ravel(), bins=num_bins_each, density=True)
            hist_ret = np.concatenate((hist_ret, hist), axis=0)
    return hist_ret

def get_image_feature_gabor(img_dir, kernels, num_crops=4):
    """
    function to extract histogram of response from gabor filter banks.
    :param img_dir: image director
    :param kernels: precomputed gabor filter bank
    :param num_crops: how many crops from the histological image
    :return: hist_ret: gabor feature vector
    """
    def compute_feats(image, kernels):
        feats = np.zeros((len(kernels), 2), dtype=np.double)
        for k, kernel in enumerate(kernels):
            filtered = ndi.convolve(image, kernel, mode='wrap')
            feats[k, 0] = filtered.mean()
            feats[k, 1] = filtered.var()
        return feats.ravel()
    image_gray = rgb2gray(rgba2rgb(Image.open(img_dir)))
    for i in range(0, num_crops):
        croped_gray, _ = random_crop_gray(image_gray, random_crop_size=(256, 256))
        if i==0:
            hist_ret = compute_feats(croped_gray, kernels)
        else:
            hist = compute_feats(croped_gray, kernels)
            hist_ret = np.concatenate((hist_ret, hist), axis=0)
    return hist_ret

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
    values = tf.reverse(values, axis=[0])
    order = tf.reverse(order, axis=[0])
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
  loss = -(edge_sum + tf.reduce_sum(tf.log(tf.sigmoid(risk_left - risk_right))* mask))/np.log(2)/edge_sum
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

all_data_csv_path = os.path.join(os.path.dirname(__file__), 'inputs', 'all_dataset.csv')
all_train_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'train')
all_test_images_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'test')
all_df = pd.read_csv(all_data_csv_path)
num_cases, _ = all_df.shape

train_image_list = os.listdir(all_train_images_path)
test_image_list = os.listdir(all_test_images_path)
some_train_img_list = []
some_test_img_list = []
train_ID_list = []
test_ID_list = []
for img_str in train_image_list:
    some_train_img_list.append(os.path.join(all_train_images_path, img_str))
    train_ID_list.append(get_imgID(img_str))

for img_str in test_image_list:
    some_test_img_list.append(os.path.join(all_test_images_path, img_str))
    test_ID_list.append(get_imgID(img_str))



all_df.set_index(keys='TCGA ID', inplace=True)
survival_val_train = []
censored_val_train = []
img_idx_list = [i for i in range(0, len(train_ID_list))]
for patient_id in train_ID_list:
    survival_val_train.append(int(all_df.loc[patient_id]['Survival months']))
    censored_val_train.append(int(all_df.loc[patient_id]['censored']))

survival_val_test = []
censored_val_test = []
for patient_id in test_ID_list:
    survival_val_test.append(int(all_df.loc[patient_id]['Survival months']))
    censored_val_test.append(int(all_df.loc[patient_id]['censored']))


# risks = tf.Variable(tf.zeros(shape=(len(survival_val_train + survival_val_test),)))
# survival_ph = tf.placeholder(dtype=tf.int32, shape=(len(survival_val_train + survival_val_test),))
# censored_ph = tf.placeholder(dtype=tf.int32, shape=(len(censored_val_train + survival_val_test),))
#
# labels = {}
# labels['survival'] = survival_ph
# labels['censored'] = censored_ph
# labels['observed'] = 1 - censored_ph
#
# # X, Y = calc_at_risk(risks, labels, len(survival_val_train + survival_val_test))
# all_losses = log_sigmoid_loss(risks, labels, len(survival_val_train+survival_val_test))
# optimizer = tf.train.GradientDescentOptimizer(0.001)
# train_var = tf.trainable_variables()
# gradients = tf.gradients(all_losses, train_var)
# train_op = optimizer.apply_gradients(zip(gradients, train_var))
# num_epochs = 20
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     prev_risks = np.zeros(shape=(len(survival_val_train + survival_val_test),))
#     for step_e in range(0, num_epochs):
#         _, logits, loss_val, survival_val_trainue, censored_value = sess.run([train_op, risks, all_losses, labels['survival'],
#                                                                         labels['censored']],
#                                                                        feed_dict={survival_ph: np.asarray(survival_val_train + survival_val_test),
#                                                                                   censored_ph: np.asarray(censored_val_train + censored_val_test)})
#         new_risks = logits
#         print('difference %1.4f'%(np.sum(np.abs(new_risks - prev_risks))))
#         prev_risks = new_risks
#         epoch_CI = eval_CI(logits, survival_val_train, censored_val_train)
#         print("loss_value is %1.4f, CI is %1.4f"%(loss_val, epoch_CI))
#     np.savetxt('estimated_risk.txt', logits)
#
#
#
#
# print('final CI is %1.4f'%(eval_CI(logits, survival_val_train, censored_val_train)))
#
# logits  = (logits - logits.min()) / (logits.max() - logits.min())

train_num_imgs = int(len(some_train_img_list))
test_num_imgs = len(some_test_img_list)

print("The number of images for training is: %d while the number of images for testing is %d"%(train_num_imgs, test_num_imgs))
kernels = []
for theta in range(4):
    theta = theta / 4. * np.pi
    for sigma in (1, 3):
        for frequency in (0.05, 0.25):
            kernel = np.real(gabor_kernel(frequency, theta=theta,
                                          sigma_x=sigma, sigma_y=sigma))
            kernels.append(kernel)


# extract image features
for idx, img_str in enumerate(some_train_img_list + some_test_img_list):
    if idx == 0:
        # img_feature_LBP = get_image_feature_LBP(img_str).reshape((1, -1))
        # img_feature_GLCM = get_image_feature_GLCM(img_str).reshape((1, -1))
        img_feature_gabor = get_image_feature_gabor(img_str, kernels).reshape((1, -1))
        img_feature_HOG = get_image_feature_HOG(img_str).reshape((1, -1))

    else:
        # img_feature_LBP = np.concatenate((img_feature_LBP, get_image_feature_LBP(img_str).reshape((1,-1))), axis=0)
        # img_feature_GLCM = np.concatenate((img_feature_GLCM, get_image_feature_GLCM(img_str).reshape((1,-1))), axis=0)
        img_feature_gabor = np.concatenate((img_feature_gabor, get_image_feature_gabor(img_str, kernels).reshape((1, -1))), axis=0)
        img_feature_HOG = np.concatenate((img_feature_HOG, get_image_feature_HOG(img_str).reshape((1, -1))), axis=0)

    if idx % 10 == 0:
        print("processing progress %1.4f %%"%(idx/len(some_train_img_list + some_test_img_list)*100))
        print(img_feature_gabor.shape)


np.savetxt('image_feature_gabor.txt', img_feature_gabor)
np.savetxt('image_feature_HOG.txt', img_feature_HOG)
# np.savetxt('image_features_LBP.txt', img_feature_LBP)
# np.savetxt('image_features_GLCM.txt', img_feature_GLCM)
# np.savetxt('survival_info.txt', np.asarray(survival_val_train + survival_val_test, dtype=np.int32))
# np.savetxt('censored.txt', np.asarray(censored_val_train + censored_val_test, dtype=np.int32))

raise ValueError("preprocessing image features finished!!!")

img_feature_LBP = np.loadtxt('image_features_LBP.txt')
img_feature_GLCM = np.loadtxt('image_features_GLCM.txt')
survival_val_train = np.loadtxt('survival_info.txt')
censored_val_train = np.loadtxt('censored.txt')
risk_logits = np.loadtxt('estimated_risk.txt')

from sklearn import preprocessing
from sklearn import linear_model
from sklearn.cross_decomposition import PLSRegression

def EvalAccuracy(y_pred, y_true):
    y_pred = y_pred.reshape((-1, 1))
    y_true = y_true.reshape((-1, 1))
    num_cases, _ = y_pred.shape
    return np.sum((y_pred - y_true)**2)/num_cases

def EvalModelAccuracy(X_train, y_train, X_test, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    print("Training Error(MSE) For %s: %f"%(model_name, EvalAccuracy(y_pred_train, y_train)))
    y_pred_test = model.predict(X_test)
    print("Testing Error(MSE) For %s: %f"%(model_name, EvalAccuracy(y_pred_test, y_test)))


X_train_LBP = img_feature_LBP[0:train_num_imgs]
X_train_LBP_uncensored = X_train_LBP[censored_val_train[0:train_num_imgs] == 0, :]
X_train_GLCM = img_feature_GLCM[0:train_num_imgs, :]
Y_train = risk_logits[0:train_num_imgs] * 1e4
Y_train_uncensored = Y_train[censored_val_train[0:train_num_imgs] == 0]
Y_train_d_inv = 1.0/(np.asarray(survival_val_train[0:train_num_imgs], dtype=np.float64)+ 1)

X_test_LBP = img_feature_LBP[train_num_imgs:train_num_imgs+test_num_imgs, :]
X_test_GLCM = img_feature_GLCM[train_num_imgs:train_num_imgs+test_num_imgs, :]
Y_test = risk_logits[train_num_imgs:train_num_imgs+test_num_imgs] * 1e4
Y_test_d_inv = 1.0/(np.asarray(survival_val_train[train_num_imgs:train_num_imgs+test_num_imgs], dtype=np.float64) + 1)


print("%d images uncensored for training"%(int(np.sum(censored_val_train[0:train_num_imgs] == 0))))
print("%d images uncensored for testing"%(int(np.sum(censored_val_train[train_num_imgs:train_num_imgs+test_num_imgs] == 0))))

print("%d images for training and %d images for testing"%(train_num_imgs, test_num_imgs))

Survival_test = survival_val_train[train_num_imgs:train_num_imgs+test_num_imgs]

Censored_test = censored_val_train[train_num_imgs:train_num_imgs+test_num_imgs]

X_test_LBP_uncensored = X_train_LBP[censored_val_train[0:train_num_imgs] == 0, :]
Survival_test_uncensored = survival_val_train[0:train_num_imgs][censored_val_train[0:train_num_imgs] == 0]
# # Baseline model: trying linear regression
# EvalModelAccuracy(X_train_LBP, Y_train, X_test_LBP, Y_test, linear_model.LinearRegression(), 'Linear Regression(LBP)')
# EvalModelAccuracy(X_train_GLCM, Y_train, X_test_GLCM, Y_test, linear_model.LinearRegression(), 'Linear Regression(GLCM)')
# # # try Bayesian Regression a.k.a ARD regression
# # EvalModelAccuracy(X_train_LBP, Y_train, X_test_LBP, Y_test, linear_model.ARDRegression(), 'ARD regression(LBP)')
# # EvalModelAccuracy(X_train_GLCM, Y_train, X_test_GLCM, Y_test, linear_model.ARDRegression(), 'ARD regression(GLCM)')
# # try Lasso
# EvalModelAccuracy(X_train_LBP, Y_train, X_test_LBP, Y_test, linear_model.Lasso(alpha=0.1), 'Lasso(LBP)')
# EvalModelAccuracy(X_train_GLCM, Y_train, X_test_GLCM, Y_test, linear_model.Lasso(alpha=0.1), 'Lasso(GLCM)')
# # try PLS regression
# EvalModelAccuracy(X_train_LBP, Y_train, X_test_LBP, Y_test, PLSRegression(n_components=X_train_LBP.shape[1]),
#                   'PLS Regression (LBP)')
# EvalModelAccuracy(X_train_GLCM, Y_train, X_test_GLCM, Y_test, PLSRegression(n_components=X_train_LBP.shape[1]),
#                   'PLS Regression (GLCM)')

# print(risk_logits[(np.asarray(survival_val_train,dtype=np.int32)>2000) & np.asarray(censored_val_train,dtype=np.bool)])

linear_regression_model_LBP = linear_model.Lasso(0.00003)
linear_regression_model_LBP.fit(X_train_LBP, Y_train)

linear_regression_model_LBP_uncensored = linear_model.Lasso(0.00003)
linear_regression_model_LBP_uncensored.fit(X_train_LBP_uncensored, Y_train_uncensored)

linear_regression_model_GLCM = linear_model.Lasso(0.00003)
linear_regression_model_GLCM.fit(X_train_GLCM, Y_train)

linear_regression_model_LBP_invd = linear_model.Lasso(alpha=0.0001)
linear_regression_model_LBP_invd.fit(X_train_LBP, survival_val_train[0:train_num_imgs])
linear_regression_model_GLCM_invd = linear_model.Lasso(alpha=0.0001)
linear_regression_model_GLCM_invd.fit(X_train_GLCM, survival_val_train[0:train_num_imgs])


test_pred_LBP = linear_regression_model_LBP.predict(X_test_LBP)
test_pred_LBP_uncensored = linear_regression_model_LBP_uncensored.predict(X_test_LBP_uncensored)
test_pred_GLCM = linear_regression_model_GLCM.predict(X_test_GLCM)

print("Using LBP features test CI is %1.4f" % (eval_CI(test_pred_LBP, Survival_test, Censored_test)))
print("Using GLCM features test CI is %1.4f" % (eval_CI(test_pred_GLCM, Survival_test, Censored_test)))

print("Using uncensored samples only obtain test CI is %1.4f" % eval_CI(test_pred_LBP_uncensored, Survival_test_uncensored, np.zeros(shape=Survival_test_uncensored.shape, dtype=np.int32)))
test_pred_LBP_invd = linear_regression_model_LBP_invd.predict(X_test_LBP)
test_pred_GLCM_invd = linear_regression_model_GLCM_invd.predict(X_test_GLCM)
print("Using LBP features fitting survival day inverse achieved CI %1.4f" % (eval_CI(1.0/test_pred_LBP_invd, Survival_test, Censored_test)))
print("Using GLCM features fitting survival day inverse achieved CI %1.4f" % (eval_CI(1.0/test_pred_GLCM_invd, Survival_test, Censored_test)))

ax1 = plt.figure(1).add_subplot(111)
ax1.scatter(survival_val_train[censored_val_train==0], risk_logits[censored_val_train==0], marker='.', edgecolor='r')
ax1.scatter(survival_val_train[censored_val_train==1], risk_logits[censored_val_train==1], marker='.', edgecolor='b')
ax1.set_xlabel('Survival Days')
ax1.set_ylabel('risk * 10^3')
ax1.legend(['not censored', 'censored'])
ax1.set_title('Estimated Risk for Patients')
plt.show()
