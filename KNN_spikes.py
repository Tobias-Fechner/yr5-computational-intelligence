# Import data processing libraries
import numpy as np

# PCA for dimensionality reduction and KNN classifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# KNN function classifier
def KNN_classifier(wavesTraining, wavesValidation, labels, n_components=4, n_neighbors=4, p=2):

    # Instantiate new PCA model, keeping only the first 10 components and fit to the training data
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

    # Create a KNN classification system using the given number of nearest neighbours to use during classification,
    # The Euclidean distance (set with p=2) is used by default
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p)
    # Fit on the normalised dataset using the given labels containing 4 unique values
    knn.fit(normalisedTraining, labels)

    # Apply trained classifier to validation data
    predictions = knn.predict(normalisedValidation)

    return predictions, componentsTraining, np.cumsum(pca.explained_variance_ratio_)
