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
SVMKernels = ['linear', 'rbf', 'poly', 'sigmoid']  # different kernels to try for kernel based SVM
Cdistance = [10**-4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 1, 10 ** 2, 10**3, 10**4] # c regularization parameter
Gammas = [10**-4, 10 ** -3, 10 ** -2, 10 ** -1, 10 ** 1, 10 ** 2, 10**3, 10**4]  # gamma regularization parameter
Degrees = [2, 3, 4, 5,10,50]  # degrees for non linear kernel methods
CVgroups = 3  # 10 cross validation groups

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

##dimensionaltiy reduction using PCA keeping only certain number of principal components ##

##find number of principal components to keep by plotting log magnitude of singular values versus singular value
#_, s, _ = np.linalg.svd(x_train_RGBhist, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('RGB Histogram Training Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('x_train_RGBhist_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(x_test_RGBhist, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('RGB Histogram Testing Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('x_test_RGBhist_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(x_test_HSVhist, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('HSV Histogram Testing Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('x_test_HSVhist_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(x_train_HSVhist, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('HSV Histogram Training Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('x_train_HSVhist_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(x_test_Labhist, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('LAB Histogram Testing Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('x_test_Labhist_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(x_train_Labhist, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('LAB Histogram Training Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('x_train_Labhist_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(xtrain_vec_g, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('Grayscale Raw Pixel Training Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('xtrain_vec_g_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(xtest_vec_g, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('Grayscale Raw Pixel Testing Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('xtest_vec_g_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(xtrain_vec, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('RGB Raw Pixel Training Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('xtrain_vec_SV.png', format='png')  # save as png
#plt.close('all')
#
#_, s, _ = np.linalg.svd(xtest_vec, full_matrices=False)
#plt.figure(figsize=(10, 7))
#plt.plot(np.linspace(0, len(s), len(s)), np.log10(s), 'ro-')
#plt.title('RGB Raw Pixel Testing Data Singular Values')
#plt.xlabel('Singular Value')
#plt.ylabel('Log10(Singular Values)')
#plt.savefig('xtest_vec_SV.png', format='png')  # save as png
#plt.close('all')

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

########################################### Implement KNN #########################################

# define grid search parameters for cross validation
knnModel = KNeighborsClassifier()  # get the class model for KNN
GridParams = {'n_neighbors': Kneigh,
              'metric': Dmetrics}  # dictionary of parameters to test and evaluate (neighbors and distances)
knn_gscv = GridSearchCV(knnModel, GridParams,
                        cv=CVgroups,
                        verbose=10)  # model for performing grid search with K and different distance metrics

# for each feature group defined above
# create feature set dictionaries
TrainFeatureSet = {'x_train_RGBhist': [x_train_RGBhist, x_test_RGBhist],
                   'x_train_HSVhist': [x_train_HSVhist, x_test_HSVhist],
                   'x_train_Labhist': [x_train_Labhist, x_test_Labhist],
                   'xtrain_vec_g': [xtrain_vec_g, xtest_vec_g],
                   'xtrain_vec': [xtrain_vec, xtest_vec]}
#for i in TrainFeatureSet:
#    # fit model to training data
#    knn_gscv.fit(TrainFeatureSet[i][0], np.ravel(y_train))
#    # get statistics for cross validation
#    StatDF = pd.DataFrame.from_dict(knn_gscv.cv_results_)
#    CVStatsFileName = i + '_trainCVResults.csv'
#    StatDF.to_csv(CVStatsFileName)  # save the cross validation statistics
#
#    # get prediction of the final test set
#    BestParams = knn_gscv.best_params_  # get the best parameters
#    knn = KNeighborsClassifier(n_neighbors=BestParams['n_neighbors'], metric=BestParams['metric'])
#    knn.fit(TrainFeatureSet[i][0], np.ravel(y_train))  # train with the best parameter on the relevant training set
#    yPred = knn.predict(TrainFeatureSet[i][1])  # test on the test set
#
#    # generate confusion matrix
#    CM = confusion_matrix(y_test, yPred)
#    df_cm = pd.DataFrame(CM, index=[i for i in labels],
#                         columns=[i for i in labels])
#    plt.figure(figsize=(12, 9))
#    sn.heatmap(df_cm, annot=True, fmt="d")
#    plt.xlabel('Predicted Class')
#    plt.ylabel('True Class')
#    CMatrixFileN = i + '_CM.png'
#    plt.savefig(CMatrixFileN, format='png')  # save as png
#    plt.close('all')
#
#    # generate and save classification report
#    report_dict = classification_report(y_test, yPred, output_dict=True)
#    reportDF = pd.DataFrame(report_dict)
#    CVStatsFileName = i + '_testCVResults.csv'
#    reportDF.to_csv(CVStatsFileName)  # save the test results
#
#


########################################### Implement SVM #########################################

GridParams = {'C': Cdistance,
              'kernel': SVMKernels, 'gamma': Gammas,
              'degrees': Degrees}  # dictionary of parameters to test and evaluate
SGDParameters = {'loss': 'hinge', 'penalty': 'l1', 'fit_intercept': False, 'verbose': 100}
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
                                          fit_intercept=False, verbose=100)
                    # train model
                    model.fit(x_train_SVM, np.ravel(y_train_SVM))
                    # test on holdout data
                    accur = model.score(x_test_SVM, np.ravel(y_test_SVM))

                    # store results for this holdout group and parameter combination
                    SVM_CVResults['params'].append({'C': c, 'kernel': kernel})
                    SVM_CVResults['scores'].append(accur)  # append accuracy score metric

                for gamma in GridParams['gamma']:
                    for degree in GridParams['degrees']:

                        if (kernel != 'linear'):
                            # transform train and test data using kernel approximation
                            feature_map_nystroem = Nystroem(gamma=gamma, degree = degree, n_components= KernelTransformationParameters['NComp'])
                            x_train_SVM_transformed = feature_map_nystroem.fit_transform(x_train_SVM)
                            x_test_SVM_transformed = feature_map_nystroem.transform(x_test_SVM) # apply same transformation to test data

                            #fit model
                            # get non-linear SVM model
                            model = SGDClassifier(loss=SGDParameters['loss'], penalty=SGDParameters['penalty'], alpha=c,
                                                  fit_intercept=False, verbose=100)
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
    #convert back to dict and store as dataframe and save to CSV file
    Results = {'Parameters': ParamsSVM[:,0], 'HO Group 1 Score': ScoresSVM[:,0],'HO Group 2 Score': ScoresSVM[:,1],'HO Group 3 Score': ScoresSVM[:,2], 'Mean CV Score': MeanScore}
    ResultsDF = pd.DataFrame(Results)
    CVStatsFileName = data + '_testCVResults_SVM.csv'
    ResultsDF.to_csv(CVStatsFileName)  # save the test results

    #Find the Best Parameter Set from Data
    MaxIndex = np.argmax(MeanScore) # get the index
    #extract the parameters
    BestParams = ParamsSVM[MaxIndex,0]

    if (BestParams['kernel'] == 'linear'): # if linear, use linear SVM Model, generate confusion matrix, and save results
        model = SGDClassifier(loss=SGDParameters['loss'], penalty=SGDParameters['penalty'], alpha=BestParams['C'],
                              fit_intercept=False, verbose=100)
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
                              fit_intercept=False, verbose=100)
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






# define grid search parameters for cross validation
#SVMModel = SVC()  # get the class model for KNN
#GridParams = {'C': Cdistance,
#              'kernel': SVMKernels, 'gamma': Gammas,
#              'degrees': Degrees}  # dictionary of parameters to test and evaluate (neighbors and distances)
#SVM_gscv = GridSearchCV(SVMModel, GridParams,
#                        cv=CVgroups, verbose=100)  # model for performing grid search with different parameters
#
## for each feature group defined above
## create feature set dictionaries
#TrainFeatureSet = {'x_train_RGBhist': [x_train_RGBhist, x_test_RGBhist],
#                   'x_train_HSVhist': [x_train_HSVhist, x_test_HSVhist],
#                   'x_train_Labhist': [x_train_Labhist, x_test_Labhist],
#                   'xtrain_vec_g': [xtrain_vec_g, xtest_vec_g],
#                   'xtrain_vec': [xtrain_vec, xtest_vec]}
#for i in TrainFeatureSet:
#    # fit model to training data
#    SVM_gscv.fit(TrainFeatureSet[i][0], np.ravel(y_train))
#    # get statistics for cross validation
#    StatDF = pd.DataFrame.from_dict(SVM_gscv.cv_results_)
#    CVStatsFileName = i + '_trainCVResults_SVM.csv'
#    StatDF.to_csv(CVStatsFileName)  # save the cross validation statistics
#
#    # get prediction of the final test set
#    BestParams = SVM_gscv.best_params_  # get the best parameters
#    SVM_model = SVC(C=BestParams['C'], kernel=BestParams['kernel'], gamma=BestParams['gamma'])
#    SVM_model.fit(TrainFeatureSet[i][0],
#                  np.ravel(y_train))  # train with the best parameter on the relevant training set
#    yPred = SVM_model.predict(TrainFeatureSet[i][1])  # test on the test set
#
#    # generate confusion matrix
#    CM = confusion_matrix(y_test, yPred)
#    df_cm = pd.DataFrame(CM, index=[i for i in labels],
#                         columns=[i for i in labels])
#    plt.figure(figsize=(10, 7))
#    sn.heatmap(df_cm, annot=True, fmt="d")
#    plt.xlabel('Predicted Class')
#    plt.ylabel('True Class')
#    CMatrixFileN = i + '_CM_SVM.png'
#    plt.savefig(CMatrixFileN, format='png')  # save as png
#    plt.close('all')
#
#    report_dict = classification_report(y_test, yPred, output_dict=True)
#    reportDF = pd.DataFrame(report_dict)
#    CVStatsFileName = i + '_testCVResults_SVM.csv'
#    reportDF.to_csv(CVStatsFileName)  # save the test results
