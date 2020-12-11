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
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold
from sklearn.linear_model import SGDClassifier
from sklearn.kernel_approximation import Nystroem
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, AveragePooling2D, MaxPooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import hinge
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers
# P = 2  # number of principal components

########################################### Data Importing and Preprocessing ##############################################
# import the training and testing data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
labels = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Adjustable Parameters
## K nearest Neighbors ##
Kneigh = [1, 3, 5, 7, 10, 25, 50, 100]  # different k values to experiment with
Dmetrics = ['manhattan', 'euclidean', 'chebyshev',
            'minkowski']
## Kernel SVM Parameters##
SVMKernels = ['linear', 'rbf', 'poly', 'sigmoid', 'laplacian']  # different kernels to try for kernel based SVM
Cdistance = [10**-8, 10**-7, 10**-6, 10 ** -5, 10 ** -4, 10**-3, 10**-2] # c regularization parameter
Gammas = [10**-8,10**-7, 10**-6, 10 ** -5, 10 ** -4, 10**-3, 10**-2]   # gamma regularization parameter
Degrees = [2,3,4,5,6,7,20]  # degrees for non linear kernel methods
CVgroups = 3  # 10 cross validation groups

## CNN Parameters ####
#KernSize = [3,5] # different kernel sizes for filter
#Padding = ['valid', 'same']
#NumFilters = [tuple([32, 64]), tuple([64]), tuple([32])] # different combinations of single layer and one combination of double layer
#PoolingFunction = ['max', 'average']
#ConvActivFunction = ['relu', 'sigmoid', 'relu', 'tanh']
#EpochSize = [100,200]
#BatchSize = [100,1000]
#Cregularizer = [10**-4, (10**-4) *.5, 10**-3] # kernel regularization


KernSize = [3] # different kernel sizes for filter
Padding = ['same']
NumFilters = [tuple([32, 64]), tuple([64]), tuple([32])] # different combinations of single layer and one combination of double layer
PoolingFunction = ['max']
ConvActivFunction = ['relu', 'sigmoid']
EpochSize = [25]
BatchSize = [500]
Cregularizer = [10**-4] # kernel regularization


#
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
    grayscale = rgb2gray(x_test[k, :, :, :]) * 255
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

# scale all data to 0 mean and unit variance before dimensionality reduction, apply transform to test data
scaler = StandardScaler()  # import standard scalar class
x_train_RGBhist = scaler.fit_transform(x_train_RGBhist)
x_test_RGBhist = scaler.transform(x_test_RGBhist)

scaler = StandardScaler()  # import standard scalar class
x_train_HSVhist = scaler.fit_transform(x_train_HSVhist)
x_test_HSVhist = scaler.transform(x_test_HSVhist)

scaler = StandardScaler()  # import standard scalar class
x_train_Labhist = scaler.fit_transform(x_train_Labhist)
x_test_Labhist = scaler.transform(x_test_Labhist)

scaler = StandardScaler()  # import standard scalar class
xtrain_vec_g = scaler.fit_transform(xtrain_vec_g)
xtest_vec_g = scaler.transform(xtest_vec_g)

scaler = StandardScaler()  # import standard scalar class
xtrain_vec = scaler.fit_transform(xtrain_vec)
xtest_vec = scaler.transform(xtest_vec)
#dimensionaltiy reduction using PCA keeping only certain number of principal components ##
#find number of principal components to keep by plotting log magnitude of singular values versus singular value
_, s, _ = np.linalg.svd(x_train_RGBhist, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('RGB Histogram Training Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('x_train_RGBhist_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(x_test_RGBhist, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('RGB Histogram Testing Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('x_test_RGBhist_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(x_test_HSVhist, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('HSV Histogram Testing Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('x_test_HSVhist_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(x_train_HSVhist, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('HSV Histogram Training Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('x_train_HSVhist_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(x_test_Labhist, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('LAB Histogram Testing Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('x_test_Labhist_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(x_train_Labhist, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('LAB Histogram Training Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('x_train_Labhist_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(xtrain_vec_g, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('Grayscale Raw Pixel Training Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('xtrain_vec_g_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(xtest_vec_g, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('Grayscale Raw Pixel Testing Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('xtest_vec_g_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(xtrain_vec, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('RGB Raw Pixel Training Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('xtrain_vec_SV.png', format='png')  # save as png
plt.close('all')

_, s, _ = np.linalg.svd(xtest_vec, full_matrices=False)
plt.figure(figsize=(10, 7))
plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
plt.title('RGB Raw Pixel Testing Data Singular Values')
plt.xlabel('Singular Value')
plt.ylabel('Log10(Singular Values)')
plt.savefig('xtest_vec_SV.png', format='png')  # save as png
plt.close('all')
P_RGBHist = 23
P_HSVHist = 35
P_LAB = 30
P_GS = 50
P_RGB = 40
pca = PCA(n_components=P_RGBHist)  # get PCA fitter
x_train_RGBhist = pca.fit_transform(x_train_RGBhist)
x_test_RGBhist = pca.transform(x_test_RGBhist)

pca = PCA(n_components=P_HSVHist)  # get PCA fitter
x_train_HSVhist = pca.fit_transform(x_train_HSVhist)
x_test_HSVhist = pca.transform(x_test_HSVhist)

pca = PCA(n_components=P_LAB)  # get PCA fitter
x_train_Labhist = pca.fit_transform(x_train_Labhist)
x_test_Labhist = pca.transform(x_test_Labhist)

pca = PCA(n_components=P_GS)  # get PCA fitter
xtrain_vec_g = pca.fit_transform(xtrain_vec_g)
xtest_vec_g = pca.transform(xtest_vec_g)

pca = PCA(n_components=P_RGB)  # get PCA fitter
xtrain_vec = pca.fit_transform(xtrain_vec)
xtest_vec = pca.transform(xtest_vec)

############################################ Implement KNN #########################################

# define grid search parameters for cross validation
knnModel = KNeighborsClassifier()  # get the class model for KNN
GridParams = {'n_neighbors': Kneigh,
              'metric': Dmetrics}  # dictionary of parameters to test and evaluate (neighbors and distances)
knn_gscv = GridSearchCV(knnModel, GridParams,
                        cv=CVgroups,
                        verbose=100)  # model for performing grid search with K and different distance metrics

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
    StatDF.to_csv(CVStatsFileName)  # save the cross validation statistics#
    # get prediction of the final test set
    BestParams = knn_gscv.best_params_  # get the best parameters
    knn = KNeighborsClassifier(n_neighbors=BestParams['n_neighbors'], metric=BestParams['metric'])
    knn.fit(TrainFeatureSet[i][0], np.ravel(y_train))  # train with the best parameter on the relevant training set
    yPred = knn.predict(TrainFeatureSet[i][1])  # test on the test set#
    # generate confusion matrix
    CM = confusion_matrix(y_test, yPred)
    df_cm = pd.DataFrame(CM, index=[i for i in labels],
                         columns=[i for i in labels])
    plt.figure(figsize=(12, 9))
    sn.heatmap(df_cm, annot=True, fmt="d")
    plt.xlabel('Predicted Class')
    plt.ylabel('True Class')
    CMatrixFileN = i + '_CM.png'
    plt.savefig(CMatrixFileN, format='png')  # save as png
    plt.close('all')#
    # generate and save classification report
    report_dict = classification_report(y_test, yPred, output_dict=True)
    reportDF = pd.DataFrame(report_dict)
    CVStatsFileName = i + '_testCVResults.csv'
    reportDF.to_csv(CVStatsFileName)  # save the test results##

########################################## Implement SVM #########################################
GridParams = {'C': Cdistance,
              'kernel': SVMKernels, 'gamma': Gammas,
              'degrees': Degrees}  # dictionary of parameters to test and evaluate
SGDParameters = {'loss': 'hinge', 'penalty': 'l1', 'fit_intercept': False, 'verbose': 1}
# use hinge loss since implementing SVM (hinge loss = linear, and for kernel method kernel transformation is applied before linear SVM)
# penalty = l1 for sparse solution
# alpha = C values defined at the top parameter list (regularizer possible values, to be determined in grid search)
# fit_intercept = False, data is already centered around 0 mean
# verbose = 10, display message updates on progress
KernelTransformationParameters = {'NComp': 100}
# 100 component approximation of training data kernel transform
# gamma = gamma values to grid search
# degree = degree values to grid search
# perform cross validation manually
for data in TrainFeatureSet:
    SVM_CVResults = {'params': [], 'scores': [],'transforms': []}  # initialize empty dictionary to store cross validation results
    SVM_TrainData = TrainFeatureSet[data][0]  # extract the training data for the relevant feature set
    ylabels = np.ravel(y_train)
    kf = KFold(n_splits=CVgroups)
    Kindices = kf.split(SVM_TrainData)  # get the generator for the indexes of train and test data
    for train_index, test_index in Kindices:
        x_train_SVM, x_test_SVM = SVM_TrainData[train_index, :], SVM_TrainData[test_index,:]  # extract data for this group of holdout and training data
        y_train_SVM, y_test_SVM = ylabels[train_index], ylabels[test_index]  # extract the labels for the holdout and training data
        ## train on training data using different hyperparameters and test on holdout data
        for kernel in GridParams['kernel']:
            for c in GridParams['C']: # calculate linear models here to avoid duplicate metrics for linear due to iterations of nonlinear parameters
                if (kernel == 'linear'):  # if linear model
                    # get linear SVM model
                    model = SGDClassifier(loss=SGDParameters['loss'], penalty=SGDParameters['penalty'], alpha=c,
                                          fit_intercept=False, verbose=SGDParameters['verbose'])
                    # train model
                    model.fit(x_train_SVM, np.ravel(y_train_SVM))
                    # test on holdout data
                    accur = model.score(x_test_SVM, np.ravel(y_test_SVM))
                    # store results for this holdout group and parameter combination
                    SVM_CVResults['params'].append({'C': c, 'kernel': kernel})
                    SVM_CVResults['scores'].append(accur)  # append accuracy score metric
                for gamma in GridParams['gamma']:
                    #degree parameter only matters for poly kernel
                    if (kernel != 'linear' and kernel != 'poly'):
                        # transform train and test data using kernel approximation
                        feature_map_nystroem = Nystroem(gamma=gamma,
                                                        n_components=KernelTransformationParameters['NComp'],
                                                        random_state=1)  # random state control so same transformation for repeated fits with same data
                        x_train_SVM_transformed = feature_map_nystroem.fit_transform(x_train_SVM)
                        x_test_SVM_transformed = feature_map_nystroem.transform(
                            x_test_SVM)  # apply same transformation to test data
                        # fit model
                        # get non-linear SVM model
                        model = SGDClassifier(loss=SGDParameters['loss'], penalty=SGDParameters['penalty'], alpha=c,
                                              fit_intercept=False, verbose=SGDParameters['verbose'])
                        # train model
                        model.fit(x_train_SVM_transformed, np.ravel(y_train_SVM))
                        # test on holdout data
                        accur = model.score(x_test_SVM_transformed, np.ravel(y_test_SVM))
                        # store results for this holdout group and parameter combination
                        SVM_CVResults['params'].append({'C': c, 'kernel': kernel, 'gamma': gamma})
                        SVM_CVResults['scores'].append(accur)  # append accuracy score metric
                        SVM_CVResults['transforms'].append(
                            feature_map_nystroem)  # append transform matrix to use for final testing set after choosing best parameters
                    for degree in GridParams['degrees']:
                        if (kernel == 'poly'):
                            # transform train and test data using kernel approximation
                            feature_map_nystroem = Nystroem(gamma=gamma, degree = degree, n_components= KernelTransformationParameters['NComp'], random_state= 1) # random state control so same transformation for repeated fits with same data
                            x_train_SVM_transformed = feature_map_nystroem.fit_transform(x_train_SVM)
                            x_test_SVM_transformed = feature_map_nystroem.transform(x_test_SVM) # apply same transformation to test data
                            #fit model
                            # get non-linear SVM model
                            model = SGDClassifier(loss=SGDParameters['loss'], penalty=SGDParameters['penalty'], alpha=c,
                                                  fit_intercept=False, verbose=SGDParameters['verbose'])
                            # train model
                            model.fit(x_train_SVM_transformed, np.ravel(y_train_SVM))
                            # test on holdout data
                            accur = model.score(x_test_SVM_transformed, np.ravel(y_test_SVM))
                            # store results for this holdout group and parameter combination
                            SVM_CVResults['params'].append({'C': c, 'kernel': kernel, 'gamma': gamma, 'degree': degree})
                            SVM_CVResults['scores'].append(accur)  # append accuracy score metric
                            SVM_CVResults['transforms'].append(feature_map_nystroem) # append transform matrix to use for final testing set after choosing best parameters
    #extract params and store as a dataframe to view cross validation results
    ParamsSVM = np.array(SVM_CVResults['params']).reshape(CVgroups,-1).T
    ScoresSVM = np.array(SVM_CVResults['scores']).reshape(CVgroups,-1).T
    TransformsSVM = np.array(SVM_CVResults['transforms'])
    MeanScore = np.mean(ScoresSVM,1) # get mean cross validation score
    # extract ranks
    RankParams = np.argsort(MeanScore)
    RankParams2 = np.copy(RankParams)
    Indices = np.linspace(len(RankParams),1, len(RankParams)) #create indices 0 - (N-1) for mapping
    RankParams2[RankParams] = Indices
    #convert back to dict and store as dataframe and save to CSV file
    Results = {'Parameters': ParamsSVM[:,0], 'HO Group 1 Score': ScoresSVM[:,0],'HO Group 2 Score': ScoresSVM[:,1],'HO Group 3 Score': ScoresSVM[:,2], 'Mean CV Score': MeanScore, 'RankScore': RankParams2}
    ResultsDF = pd.DataFrame(Results)
    CVStatsFileName = data + '_testCVResults_SVM.csv'
    ResultsDF.to_csv(CVStatsFileName)  # save the test results
    #Find the Best Parameter Set from Data
    MaxIndex = np.argmax(MeanScore) # get the index
    #extract the parameters
    BestParams = ParamsSVM[MaxIndex,0]
    if (BestParams['kernel'] == 'linear'): # if linear, use linear SVM Model, generate confusion matrix, and save results
        model = SGDClassifier(loss=SGDParameters['loss'], penalty=SGDParameters['penalty'], alpha=BestParams['C'],
                              fit_intercept=False, verbose=SGDParameters['verbose'])
        SVM_TestDataFinal = TrainFeatureSet[data][1]
        model.fit(TrainFeatureSet[data][0], np.ravel(y_train)) # fit model on training data with best parameters
        yPred = model.predict(SVM_TestDataFinal)
         #generate confusion matrix
        CM = confusion_matrix(y_test, yPred)
        df_cm = pd.DataFrame(CM, index=[i for i in labels],
                             columns=[i for i in labels])
        plt.figure(figsize=(12, 9))
        sn.heatmap(df_cm, annot=True, fmt="d")
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        CMatrixFileN = data + '_CM_SVM.png'
        plt.savefig(CMatrixFileN, format='png')  # save as png
        plt.close('all')
        # generate and save classification report
        report_dict = classification_report(y_test, yPred, output_dict=True)
        reportDF = pd.DataFrame(report_dict)
        CVStatsFileName = data + '_testResults_SVM.csv'
        reportDF.to_csv(CVStatsFileName)  # save the test results
    else:
        #get transformation to apply on testing data
        TransformExtractIndex = MaxIndex - len(GridParams['C'])
        ExtractedTransform = TransformsSVM[TransformExtractIndex]
        #perform same transformation as was on training data
        SVM_TestDataFinal = TrainFeatureSet[data][1]
        SVM_TrainData_Transformed = ExtractedTransform.transform(TrainFeatureSet[data][0])
        SVM_TestDataFinal_Transformed = ExtractedTransform.transform(SVM_TestDataFinal) # apply same transformation to test data
        model = SGDClassifier(loss=SGDParameters['loss'], penalty=SGDParameters['penalty'], alpha=BestParams['C'],
                              fit_intercept=False, verbose=SGDParameters['verbose'])
        model.fit(SVM_TrainData_Transformed, np.ravel(y_train)) # fit model on training data with best parameters
        yPred = model.predict(SVM_TestDataFinal_Transformed)
        # generate confusion matrix
        CM = confusion_matrix(y_test, yPred)
        df_cm = pd.DataFrame(CM, index=[i for i in labels],
                             columns=[i for i in labels])
        plt.figure(figsize=(12, 9))
        sn.heatmap(df_cm, annot=True, fmt="d")
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        CMatrixFileN = data + '_CM_SVM.png'
        plt.savefig(CMatrixFileN, format='png')  # save as png
        plt.close('all')
        # generate and save classification report
        report_dict = classification_report(y_test, yPred, output_dict=True)
        reportDF = pd.DataFrame(report_dict)
        CVStatsFileName = data + '_testResults_SVM.csv'
        reportDF.to_csv(CVStatsFileName)  # save the test results




###################################### Implement Convolutional Neural Network ########################################
GridParams = {'KernSize': KernSize,
              'padding': Padding, 'ActivFunction': ConvActivFunction,
              'NumFilters': NumFilters, 'Pooling': PoolingFunction, 'BatchSize': BatchSize, 'EpochSize': EpochSize, 'C': Cregularizer}  # dictionary of parameters to test and evaluate
#opt = SGD(learning_rate=0.01)
opt = 'adam' # better optimizer
Early_Stop = EarlyStopping(monitor='loss', patience=5) #early stop criteria from keras when error not improve or doesnt change after many epochs
#opt = SGD(learning_rate=0.01)#get optimizer
verbose1 = 0
## PREPROCESSING ##
##convert images to the right shape and color distribution, build the dataset dictionary
# intialize storage matrices
x_test_RGB_CNN = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[1], 3))  #RGB images, test
x_train_RGB_CNN = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[1], 3)) # RGB images, train
x_test_HSV_CNN = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[1], 3))  #HSV images, test
x_train_HSV_CNN = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[1], 3))  # HSV images, train
x_test_Lab_CNN = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[1], 3))  # LAB images, test
x_train_Lab_CNN = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[1], 3))  # LAB images, test
x_test_gray_CNN = np.zeros((x_test.shape[0], x_test.shape[1], x_test.shape[1], 1)) # Gray images, test
x_train_gray_CNN = np.zeros((x_train.shape[0], x_train.shape[1], x_train.shape[1], 1)) # gray images, train
#Normalization methods for different colorspaces
#1.0 * lab.L / 100,
#1.0 * (lab.A + 86.185) / 184.439,
#1.0 * (lab.B + 107.863) / 202.345);

### Extract colorspace images and normalize
for k in range(0, x_test.shape[0]):

    hsvImage = rgb2hsv(x_test[k, :, :, :])  # convert to HSV colorspace
    labImage = rgb2lab(x_test[k, :, :, :])
    labImage[:,:,0] =  1.0 * labImage[:,:,0] / 100 # normalize lab image (HSV and grayscale already normalized)
    labImage[:,:,1] =  1.0 * (labImage[:,:,1] + 86.185) / 184.439 # normalize lab image (HSV and grayscale already normalized)
    labImage[:,:,2] =  1.0 * (labImage[:,:,2] + 107.863) / 202.345 # normalize lab image (HSV and grayscale already normalized)
    RGBImage = x_test[k,:,:,:] / 255 # normalize RGB image
    grayscale = rgb2gray(x_test[k, :, :, :])
    x_test_gray_CNN[k, :, :,0] = grayscale
    x_test_HSV_CNN[k, :, :,:] = hsvImage
    x_test_Lab_CNN[k, :, :,:] = labImage
    x_test_RGB_CNN[k, :, :,:] = RGBImage

for k in range(0, x_train.shape[0]):
    hsvImage = rgb2hsv( x_train[k, :, :, :])  # convert to HSV colorspace
    labImage = rgb2lab( x_train[k, :, :, :])
    labImage[:, :, 0] = 1.0 * labImage[:, :, 0] / 100  # normalize lab image (HSV and grayscale already normalized)
    labImage[:, :, 1] = 1.0 * (labImage[:, :, 1] + 86.185) / 184.439  # normalize lab image (HSV and grayscale already normalized)
    labImage[:, :, 2] = 1.0 * (labImage[:, :, 2] + 107.863) / 202.345  # normalize lab image (HSV and grayscale already normalized)
    RGBImage =  x_train[k, :, :, :] /255
    grayscale = rgb2gray( x_train[k, :, :, :])
    x_train_gray_CNN[k, :, :, 0] = grayscale
    x_train_HSV_CNN[k, :, :, :] = hsvImage
    x_train_Lab_CNN[k, :, :, :] = labImage
    x_train_RGB_CNN[k, :, :, :] = RGBImage


TrainFeatureSetCNN = {'x_train_RGB_CNN': [x_train_RGB_CNN, x_test_RGB_CNN],
                   'x_train_HSV_CNN': [x_train_HSV_CNN, x_test_HSV_CNN],
                   'x_train_Lab_CNN': [x_train_Lab_CNN, x_test_Lab_CNN],
                   'xtrain_vec_g_CNN': [x_train_gray_CNN, x_test_gray_CNN]}

## One Hot Encoding
#y_trainCat = to_categorical(y_train)
#y_testCat = to_categorical(y_test)


# perform cross validation manually
for data in TrainFeatureSetCNN:

    CNN_CV_Results = {'params': [], 'scores': [],'transforms': []}  # initialize empty dictionary to store cross validation results

    CNN_TrainData = TrainFeatureSetCNN[data][0]  # extract the training data for the relevant feature set
    ylabels = np.ravel(y_train)
    kf = KFold(n_splits=CVgroups)
    Kindices = kf.split(CNN_TrainData)  # get the generator for the indexes of train and test data

    for train_index, test_index in Kindices:
        x_train_CNN, x_test_CNN = CNN_TrainData[train_index, :], CNN_TrainData[test_index,:]  # extract data for this group of holdout and training data
        y_train_CNN, y_test_CNN = ylabels[train_index], ylabels[test_index]  # extract the labels for the holdout and training data

        ## train on training data using different hyperparameters and test on holdout data

        for kernel in GridParams['KernSize']:
            for padding in GridParams['padding']: # calculate linear models here to avoid duplicate metrics for linear due to iterations of nonlinear parameters
                for  AcFunc in GridParams['ActivFunction']:
                    for pool in GridParams['Pooling']:
                        for C in GridParams['C']:
                            for BS in GridParams['BatchSize']:
                                for ES in GridParams['EpochSize']:
                                    for NF in GridParams['NumFilters']:

                                        ## build neural network with the specified parameters
                                        model = Sequential()
                                        if (len(NF) != 2): # if one layer model, build one hidden layer model

                                            model.add(Conv2D(NF[0], kernel, activation= AcFunc, input_shape = np.shape(x_train_CNN[0,:,:,:]), kernel_regularizer=regularizers.L2(C))) # use shape of first image as expected input

                                            if(pool == 'max'):
                                                model.add(MaxPooling2D(pool_size= [2,2], strides=(2,2), padding=padding)) # add pooling layer
                                            else:
                                                model.add(AveragePooling2D(pool_size= [2,2], strides=(2,2), padding=padding)) # add pooling layer

                                            model.add(Conv2D(64, kernel, activation=AcFunc, kernel_regularizer=regularizers.L2(C)))  # add second layer with specified number of neurons
                                            model.add(Flatten()) # add flat layer
                                            model.add(Dense(64, activation='relu'))
                                            model.add(Dense(10,activation = 'softmax')) # output linear classification

                                            #compile model using specified batch and epoch size
                                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=opt, metrics=['accuracy'])
                                            # provide validation hold out data to get periodic updates on performance of hold out(no training)
                                            model.fit(x_train_CNN, y_train_CNN, batch_size=BS, epochs=ES, verbose=verbose1, validation_data=(x_test_CNN, (y_test_CNN)), callbacks=Early_Stop)
                                            score = model.evaluate(x_test_CNN, (y_test_CNN), verbose=verbose1)
                                            accur = score[1]  # get the accuracy to store for holdout data

                                            # store results for this holdout group and parameter combination
                                            CNN_CV_Results['params'].append({'kernel Size': kernel, 'padding': padding, 'AcFunc': AcFunc, 'pool': pool, 'Batch Size': BS, 'Epoch Size': ES, 'C': C, 'Num Nodes': NF})
                                            CNN_CV_Results['scores'].append(accur)  # append accuracy score metric
                                        else:
                                            model.add(Conv2D(NF[0], kernel, activation= AcFunc, input_shape = np.shape(x_train_CNN[0,:,:,:]), kernel_regularizer=regularizers.L2(C))) # use shape of first image as expected input

                                            if(pool == 'max'):
                                                model.add(MaxPooling2D(pool_size= [2,2], strides=(2,2), padding=padding)) # add pooling layer
                                            else:
                                                model.add(AveragePooling2D(pool_size= [2,2], strides=(2,2), padding=padding)) # add pooling layer

                                            model.add(Conv2D(NF[1], kernel, activation= AcFunc, kernel_regularizer=regularizers.L2(C))) # add second layer with specified number of neurons
                                            if (pool == 'max'):
                                                model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2),
                                                                       padding=padding))  # add pooling layer
                                            else:
                                                model.add(AveragePooling2D(pool_size=[2, 2], strides=(2, 2),padding=padding))  # add pooling layer

                                            model.add(Conv2D(64, kernel, activation=AcFunc,kernel_regularizer=regularizers.L2(C)))  # add second layer with specified number of neurons
                                            model.add(Flatten()) # add flat layer
                                            model.add(Dense(64, activation='relu'))
                                            model.add(Dense(10,activation = 'softmax')) # output linear classification

                                            #compile model using specified batch and epoch size
                                            model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=opt, metrics=['accuracy'])
                                            # provide validation hold out data to get periodic updates on performance of hold out(no training)
                                            model.fit(x_train_CNN, y_train_CNN, batch_size=BS, epochs=ES, verbose=verbose1, validation_data=(x_test_CNN, (y_test_CNN)), callbacks= Early_Stop)
                                            score = model.evaluate(x_test_CNN, (y_test_CNN), verbose=verbose1)
                                            accur = score[1] # get the accuracy to store for holdout data

                                            # store results for this holdout group and parameter combination
                                            CNN_CV_Results['params'].append({'kernel Size': kernel, 'padding': padding, 'AcFunc': AcFunc, 'pool': pool, 'Batch Size': BS, 'Epoch Size': ES, 'C': C, 'Num Nodes': NF})
                                            CNN_CV_Results['scores'].append(accur)  # append accuracy score metric

    # extract params and store as a dataframe to view cross validation results
    ParamsSVM = np.array(CNN_CV_Results['params']).reshape(CVgroups, -1).T
    ScoresSVM = np.array(CNN_CV_Results['scores']).reshape(CVgroups, -1).T
    MeanScore = np.mean(ScoresSVM, 1)  # get mean cross validation score
    # extract ranks
    RankParams = np.argsort(MeanScore)
    RankParams2 = np.copy(RankParams)
    Indices = np.linspace(len(RankParams), 1, len(RankParams))  # create indices 0 - (N-1) for mapping
    RankParams2[RankParams] = Indices
    # convert back to dict and store as dataframe and save to CSV file
    Results = {'Parameters': ParamsSVM[:, 0], 'HO Group 1 Score': ScoresSVM[:, 0], 'HO Group 2 Score': ScoresSVM[:, 1],
               'HO Group 3 Score': ScoresSVM[:, 2], 'Mean CV Score': MeanScore, 'RankScore': RankParams2}
    ResultsDF = pd.DataFrame(Results)
    CVStatsFileName = data + '_CVResults_CNN.csv'
    ResultsDF.to_csv(CVStatsFileName)  # save the CV test results

    # Find the Best Parameter Sets from Data
    MaxIndex = np.argmax(MeanScore)  # get the index
    # extract the parameters
    BestParams = ParamsSVM[MaxIndex, 0]

    #train and test on the given test data with optimized parameters
    model = Sequential()
    CNN_TestData = TrainFeatureSetCNN[data][1] # get the testing data
    if (len(BestParams['Num Nodes']) == 2): #if best network was two layers, build the two layer model
        model.add(Conv2D(BestParams['Num Nodes'][0], BestParams['kernel Size'], activation=BestParams['AcFunc'], input_shape=np.shape(CNN_TrainData[0, :, :, :]), kernel_regularizer= regularizers.L2(BestParams['C'])))  # use shape of first image as expected input
        if (BestParams['pool'] == 'max'):
            model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding=BestParams['padding']))  # add pooling layer
        else:
            model.add(AveragePooling2D(pool_size=[2, 2], strides=(2, 2), padding=BestParams['padding']))  # add pooling layer
        model.add(Conv2D(BestParams['Num Nodes'][1], BestParams['kernel Size'], activation=BestParams['AcFunc'], kernel_regularizer= regularizers.L2(BestParams['C'])))  # add second layer with specified number of neurons
        if (BestParams['pool'] == 'max'):
            model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding=BestParams['padding']))  # add pooling layer
        else:
            model.add(AveragePooling2D(pool_size=[2, 2], strides=(2, 2), padding=BestParams['padding']))  # add pooling layer

        model.add(Conv2D(64, BestParams['kernel Size'], activation=BestParams['AcFunc'],kernel_regularizer=regularizers.L2(BestParams['C'])))  # add second layer with specified number of neurons
        model.add(Flatten())  # add flat layer
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))  # output linear classification

        # compile model using specified batch and epoch size
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),optimizer=opt, metrics=['accuracy'])

        ##fit model
        model.fit(CNN_TrainData, (y_train), batch_size=BestParams['Batch Size'], epochs=BestParams['Epoch Size'], verbose=verbose1,callbacks=Early_Stop) # train on all of the training data with the most optimized parameters
        yhat = model.predict(CNN_TestData, verbose=verbose1)
        yhat = np.argmax(yhat, axis = 1)
        # generate confusion matrix
        CM = confusion_matrix(y_test, yhat)
        df_cm = pd.DataFrame(CM, index=[i for i in labels],
                             columns=[i for i in labels])
        plt.figure(figsize=(12, 9))
        sn.heatmap(df_cm, annot=True, fmt="d")
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        CMatrixFileN = data + '_CM_CNN.png'
        plt.savefig(CMatrixFileN, format='png')  # save as png
        plt.close('all')

        # generate and save classification report
        report_dict = classification_report(y_test, yhat, output_dict=True)
        reportDF = pd.DataFrame(report_dict)
        CVStatsFileName = data + '_testResults_CNN.csv'
        reportDF.to_csv(CVStatsFileName)  # save the test results


    else:
        model.add( Conv2D(BestParams['Num Nodes'][0], BestParams['kernel Size'], activation=BestParams['AcFunc'],input_shape=np.shape(CNN_TrainData[0, :, :, :]), kernel_regularizer= regularizers.L2(BestParams['C'])))  # use shape of first image as expected input
        if (BestParams['pool'] == 'max'):
            model.add(MaxPooling2D(pool_size=[2, 2], strides=(2, 2), padding=BestParams['padding']))  # add pooling layer
        else:
            model.add(AveragePooling2D(pool_size=[2, 2], strides=(2, 2), padding=BestParams['padding']))  # add pooling layer
        model.add(Conv2D(64, BestParams['kernel Size'], activation=BestParams['AcFunc'], kernel_regularizer=regularizers.L2(BestParams['C'])))  # add second layer with specified number of neurons
        model.add(Flatten())  # add flat layer
        model.add(Dense(64, activation='relu'))
        model.add(Dense(10, activation='softmax'))  # output linear classification

        # compile model using specified batch and epoch size
        #opt = SGD(learning_rate=0.01)  # get optimizer
        model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=opt,metrics=['accuracy'])
        model.fit(CNN_TrainData, (y_train), batch_size=BestParams['Batch Size'], epochs=BestParams['Epoch Size'],
                  verbose=verbose1, callbacks=Early_Stop)  # train on all of the training data with the most optimized parameters
        yhat = model.predict(CNN_TestData, verbose=verbose1)
        yhat = np.argmax(yhat, axis=1)
        # generate confusion matrix
        CM = confusion_matrix(y_test, yhat)
        df_cm = pd.DataFrame(CM, index=[i for i in labels],
                             columns=[i for i in labels])
        plt.figure(figsize=(12, 9))
        sn.heatmap(df_cm, annot=True, fmt="d")
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        CMatrixFileN = data + '_CM_CNN.png'
        plt.savefig(CMatrixFileN, format='png')  # save as png
        plt.close('all')

        # generate and save classification report
        report_dict = classification_report(y_test, yhat, output_dict=True)
        reportDF = pd.DataFrame(report_dict)
        CVStatsFileName = data + '_testResults_CNN.csv'
        reportDF.to_csv(CVStatsFileName)  # save the test results

