import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

def pcaKnn(wavesTraining, wavesValidation, labels, n_components=8, n_neighbors=6, p=2):
    """
    Function extracts principle components by fitting sklearn's PCA model to the training waveform extracts dataset,
    using n_components specified with a configurable parameter.
    :param wavesTraining: dataset of spike waveform extracts used for training
    :param wavesValidation: dataset of spike waveform extracts used for validation
    :param labels: dataset of spike class labels used for training the KNN
    :param n_components: number of components to retain from the principle component analysis
    :param n_neighbors: number of neighbours to evaluate to classify an incoming waveform sample during the K-nearest neighbour
    classification
    :param p: configurable for the KNeighborsClassifier to select the kind of distance used to evaluate the nearest neighbours. Default to
    2, denoting the Euclidean (straight line) distance
    :return: return a tuple of the predicted classes, the components extracted from the training data, and a cumulative sum of the explained
    variance for the data the pca model is fitted to.
    """

    # Instantiate new PCA model, keeping only the first X components (specified by n_components) and fit to the training data
    pca = PCA(n_components=n_components)
    pca.fit(wavesTraining)

    # Print the total variance explained
    print("Total Variance Explained: ", np.sum(pca.explained_variance_ratio_))

    # Extract the principal components from the training data and transform the validation data using those components
    componentsTraining = pca.fit_transform(wavesTraining)
    componentsValidation = pca.transform(wavesValidation)

    # Normalise the datasets
    min_max_scaler = MinMaxScaler()
    normalisedTraining = min_max_scaler.fit_transform(componentsTraining)
    normalisedValidation = min_max_scaler.fit_transform(componentsValidation)

    # Create a KNN classification system using the given number of nearest neighbours to use during classification
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
    # Fit on the normalised dataset using the given labels containing 4 unique values (classes)
    knn.fit(normalisedTraining, labels)

    # Apply trained classifier to validation data
    predictions = knn.predict(normalisedValidation)

    return predictions, componentsTraining, np.cumsum(pca.explained_variance_ratio_)
