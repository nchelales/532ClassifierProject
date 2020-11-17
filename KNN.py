# KNN, classifier model for classifying an color image of one of 10 classes in the cifar-10 dataset

def KNN(X, Xtrain, Ytrain, K, D):
    # X, the data to be classified NxMx3 image
    # Xtrain, the training data used to get the nearest neighbors LxMxNx3
    # Ytrain, the labels associated with the training data Lx1
    # K, the number of nearest neighbors to use, scalar
    # D, the distance metric to use, scalar mapping variable (Dmetrics)

    import numpy as np
    from scipy import stats
    from scipy import spatial

    # distance metrics setup
    Dmetrics = ['cityblock', 'euclidean', 'cosine',
                'mahalanobis']  # manhattan, euclidean, cosine, and Mahalanobis distance
    D_i = Dmetrics[D]  # get the correct distance metric
    D_images = np.zeros((Xtrain.shape[0],1))  # Lx1 distance metric for each image in training

    # vectorize the images to be a vector of of each color layer, continuous  NoPixels * 3 x 1
    X = np.reshape(np.transpose(X, tuple([2, 0, 1])), (X.shape[0] * X.shape[1] * X.shape[2], 1))
    Xtrain = np.reshape(
        np.transpose(Xtrain, tuple([0, 3, 1, 2])),
        (Xtrain.shape[0], Xtrain.shape[1] * Xtrain.shape[2] * Xtrain.shape[3], 1))

    for L in range(0, Xtrain.shape[0]):  # iterate through images, calculating distance metric
        D_calc = np.mean(np.diag(spatial.distance.cdist(X[:], Xtrain[L, :], D_i)))
        D_images[L, 0] = D_calc  # store the distance using the given metric


    # classes of the K nearest neighbors
    Dsorted = np.argsort(D_images, axis=0)
    Kneighbors = Dsorted[0:K, 0]  # get the kneighbors indices
    Kneighbors = Ytrain[Kneighbors]  # extract the classes of the neighbors

    #get the class estimate
    Yhat = stats.mode(Kneighbors)
    Yhat = Yhat[0][0]

    return Yhat
