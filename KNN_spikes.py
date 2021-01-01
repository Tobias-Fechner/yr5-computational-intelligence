# Import data processing libraries
import numpy as np
import pandas as pd

# PCA for dimensionality reduction and KNN classifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# KNN function classifier
def KNN_classifier(data_training, data_validation, labels):

    # Instantiate new PCA model with 4 components, and fit to the training data
    pca = PCA(n_components=4)
    pca.fit(data_training)

    # Print the total variance explained
    print("Total Variance Explained: ", np.sum(pca.explained_variance_ratio_))

    # Extract the principal components from the training data and transform the validation data using those components
    componentsTraining = pca.fit_transform(data_training)
    componentsValidation = pca.transform(data_validation)

    # Normalise the datasets
    min_max_scaler = MinMaxScaler()
    normalisedTraining = min_max_scaler.fit_transform(componentsTraining)
    normalisedValidation = min_max_scaler.fit_transform(componentsValidation)

    # Create a KNN classification system with k = 4, using the (p2) Euclidean norm and fit on the training data
    knn = KNeighborsClassifier(n_neighbors=4, p=2)
    knn.fit(normalisedTraining, labels)

    # Apply trained classifier to validation data
    predictions = knn.predict(normalisedValidation)

    return predictions, componentsTraining
