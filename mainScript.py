import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from skimage.color import rgb2gray
from skimage.exposure import histogram
from skimage.color import rgb2hsv
from skimage.color import rgb2lab
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

P = 2  # number of principal components

# import the training and testing data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
## K nearest Neighbors ##
# Adjustable Parameters
Kneigh = [1, 3, 5, 10, 25, 50, 100, 1000, 5000, 10000]  # different k values to experiment with
Dmetrics = ['manhattan', 'euclidean', 'chebyshev',
            'minkowski']
CVgroups = 10  # 10 cross validation groups

### Feature Extraction ###
# Feature Sets: Raw Pixel values (vectorized images, in RGB)
#               Grayscale images (vectorized image)
# RGB colorspace histogram data
# HSV colorspace histogram data
# Lab colorspace histogram data

# intialize storage matrices
x_test_RGBhist = np.zeros((x_test.shape[0], 256 * 3))  # vectorized color histogram
x_train_RGBhist = np.zeros((x_train.shape[0], 256 * 3))  # vectorized color histogram
x_test_HSVhist = np.zeros((x_test.shape[0], 256 * 3))  # vectorized color histogram
x_train_HSVhist = np.zeros((x_train.shape[0], 256 * 3))  # vectorized color histogram
x_test_Labhist = np.zeros((x_test.shape[0], 256 * 3))  # vectorized color histogram
x_train_Labhist = np.zeros((x_train.shape[0], 256 * 3))  # vectorized color histogram
x_test_gray = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[2]))
x_train_gray = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[2]))
scaler = StandardScaler()  # import standard scalar class

### Extract color histograms and grayscale images ###
for k in range(0, x_test.shape[0]):

    hsvImage = rgb2hsv(x_test[k, :, :, :])  # convert to HSV colorspace
    labImage = rgb2lab(x_test[k, :, :, :])
    for layer in range(0, 3):
        imLayer = x_test[k, :, :, layer]  # get the image layer
        hsvLayer = hsvImage[:, :, layer]  # get the image layer
        labLayer = labImage[:, :, layer]  # get the image layer
        hist, _ = histogram(imLayer.astype(float), nbins=256)  # get the RGB histogram of that color layer
        x_test_RGBhist[k, layer * (256): layer * (256) + 256] = hist  # store the histogram
        hist, _ = histogram(hsvLayer.astype(float), nbins=256)  # get the HSV histogram of that color layer
        x_test_HSVhist[k, layer * (256): layer * (256) + 256] = hist  # store the histogram
        hist, _ = histogram(labLayer.astype(float), nbins=256)  # get the LAB histogram of that color layer
        x_test_Labhist[k, layer * (256): layer * (256) + 256] = hist  # store the histogram

    # convert test image to grayscale and store it
    grayscale = rgb2gray(x_test[k, :, :, :])
    x_test_gray[k, :, :] = grayscale

## Repeat for train images
### Extract color histograms and grayscale images ###
for k in range(0, x_train.shape[0]):
    hsvImage = rgb2hsv(x_train[k, :, :, :])  # convert to HSV colorspace
    labImage = rgb2lab(x_train[k, :, :, :])
    for layer in range(0, 3):
        imLayer = x_train[k, :, :, layer]  # get the image layer
        hsvLayer = hsvImage[:, :, layer]  # get the image layer
        labLayer = labImage[:, :, layer]  # get the image layer
        hist, _ = histogram(imLayer.astype(float), nbins=256)  # get the RGB histogram of that color layer
        x_train_RGBhist[k, layer * (256): layer * (256) + 256] = hist  # store the histogram
        hist, _ = histogram(hsvLayer.astype(float), nbins=256)  # get the HSV histogram of that color layer
        x_train_HSVhist[k, layer * (256): layer * (256) + 256] = hist  # store the histogram
        hist, _ = histogram(labLayer.astype(float), nbins=256)  # get the LAB histogram of that color layer
        x_train_Labhist[k, layer * (256): layer * (256) + 256] = hist  # store the histogram

    # convert test image to grayscale and store it
    grayscale = rgb2gray(x_train[k, :, :, :]) * 255
    x_train_gray[k, :, :] = grayscale

# vectorize the RGB images for the final possible feature set
xtrain_vec = np.squeeze(np.reshape(
    np.transpose(x_train, tuple([0, 3, 1, 2])),
    (x_train.shape[0], x_train.shape[1] * x_train.shape[2] * x_train.shape[3], 1)))
xtest_vec = np.squeeze(np.reshape(
    np.transpose(x_test, tuple([0, 3, 1, 2])),
    (x_test.shape[0], x_test.shape[1] * x_test.shape[2] * x_test.shape[3], 1)))
xtrain_vec_g = np.squeeze(
    np.reshape(x_train_gray, (x_train_gray.shape[0], x_train_gray.shape[1] * x_train_gray.shape[2], 1)))
xtest_vec_g = np.squeeze(
    np.reshape(x_test_gray, (x_test_gray.shape[0], x_test_gray.shape[1] * x_test_gray.shape[2], 1)))

# scale all data to 0 mean and unit variance before dimensionality reduction
x_train_RGBhist = scaler.fit_transform(x_train_RGBhist)
x_test_RGBhist = scaler.fit_transform(x_test_RGBhist)
x_test_HSVhist = scaler.fit_transform(x_test_HSVhist)
x_train_HSVhist = scaler.fit_transform(x_train_HSVhist)
x_test_Labhist = scaler.fit_transform(x_test_Labhist)
x_train_Labhist = scaler.fit_transform(x_train_Labhist)
xtrain_vec_g = scaler.fit_transform(xtrain_vec_g)
xtest_vec_g = scaler.fit_transform(xtest_vec_g)
xtrain_vec = scaler.fit_transform(xtrain_vec)
xtest_vec = scaler.fit_transform(xtest_vec)

##dimensionaltiy reduction using PCA keeping only two principal components ##
pca = PCA(n_components=P)  # get PCA fitter
x_train_RGBhist = pca.fit_transform(x_train_RGBhist)
x_test_RGBhist = pca.fit_transform(x_test_RGBhist)
x_test_HSVhist = pca.fit_transform(x_test_HSVhist)
x_train_HSVhist = pca.fit_transform(x_train_HSVhist)
x_test_Labhist = pca.fit_transform(x_test_Labhist)
x_train_Labhist = pca.fit_transform(x_train_Labhist)
xtrain_vec_g = pca.fit_transform(xtrain_vec_g)
xtest_vec_g = pca.fit_transform(xtest_vec_g)
xtrain_vec = pca.fit_transform(xtrain_vec)
xtest_vec = pca.fit_transform(xtest_vec)

# define grid search parameters for cross validation
knnModel = KNeighborsClassifier()  # get the class model for KNN
GridParams = {'n_neighbors': Kneigh,
              'metric': Dmetrics}  # dictionary of parameters to test and evaluate (neighbors and distances)
knn_gscv = GridSearchCV(knnModel, GridParams,
                        cv=CVgroups)  # model for performing grid search with K and different distance metrics

# for each feature group defined above
# create feature set dictionaries
TrainFeatureSet = {'x_train_RGBhist': [x_train_RGBhist, x_test_RGBhist],
                   'x_train_HSVhist': [x_train_HSVhist, x_test_HSVhist],
                   'x_train_Labhist': [x_train_Labhist, x_test_Labhist],
                   'xtrain_vec_g': [xtrain_vec_g, xtest_vec_g],
                   'xtrain_vec': [xtrain_vec, xtest_vec]}
for i in TrainFeatureSet:
    # fit model to training data
    knn_gscv.fit(TrainFeatureSet[i][0], np.ravel(y_train))
    # get statistics for cross validation
    StatDF = pd.DataFrame.from_dict(knn_gscv.cv_results_)
    CVStatsFileName = i + '_trainCVResults.csv'
    StatDF.to_csv(CVStatsFileName)  # save the cross validation statistics

    # get prediction of the final test set
    BestParams = knn_gscv.best_params_ #get the best parameters
    knn = KNeighborsClassifier(n_neighbors=BestParams['n_neighbors'], metric=BestParams['metric'])
    knn.fit(TrainFeatureSet[i][0], np.ravel(y_train)) # train with the best parameter on the relevant training set
    yPred = knn.predict(TrainFeatureSet[i][1]) # test on the test set

    # generate confusion matrix
    CM = confusion_matrix(y_test, yPred)
    df_cm = pd.DataFrame(CM, index=[i for i in labels],
                         columns=[i for i in labels])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    CMatrixFileN = i + '_CM.png'
    plt.savefig(CMatrixFileN, format ='png') # save as png
    plt.close('all')




