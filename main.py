import numpy as np
from sklearn.datasets import fetch_openml
from PCA import PCA

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')
X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# Normalize the input data
X /= 255.

# Split the dataset into training and testing sets
X_train, y_train = X[:60000], y[:60000]
X_test, y_test = X[60000:], y[60000:]

# Reduce the dimensionality of the input data using PCA
n_components = 50
pca = PCA(n_components=n_components)
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Train a k-means clustering model on the reduced data
from sklearn.cluster import KMeans
k = 10
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_train_pca)

# Test the classifier on a new record
# (Assuming the new record is stored as a 1D array 'new_record')
print(type(X_test))
new_record = X_test[0]  # Change to any other image in the dataset
new_record = np.array(new_record)
new_record = new_record.reshape(1, -1)
new_reduced = pca.transform(new_record)
predicted_cluster = kmeans.predict(new_reduced)[0]

# Print the predicted cluster and actual label of the new record
print("Predicted cluster:", predicted_cluster)
print("Actual label:", y_test[0])
