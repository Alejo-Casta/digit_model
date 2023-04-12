from sklearn.datasets import fetch_openml
from PCA import PCA
from TSNE import TSNE
from SVD import SVD
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load the MNIST dataset
mnist = fetch_openml('mnist_784')

X = mnist.data.astype('float32')
y = mnist.target.astype('int64')

# mnist.target = mnist.target.astype(int)
# X = mnist.data[(mnist.target == 0) | (mnist.target == 8)]
# y = mnist.target[(mnist.target == 0) | (mnist.target == 8)]

# Normalize the input data
X /= 255.

# X_train, y_train = X[:10000], y[:10000]
# X_test, y_test = X[10000:12000], y[10000:12000]

method = "SVD"

if method == "PCA":
    # Dimensionality reduction using PCA
    pca = PCA(n_components=50)
    pca.fit(X)
    X_train_pca = pca.fit_transform(X)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_train_pca, y, test_size=0.2, random_state=42)
    # X_test_pca = pca.fit_transform(X_test)

elif method == "SVD":
    # Dimensionality reduction using SVD
    svd = SVD(n_components=50)
    prueba = svd.fit(X)
    X_train_pca = svd.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_train_pca, y, test_size=0.2, random_state=42)
    # X_test_pca = svd.fit_transform(X_test)

elif method == "t-SNE":
    # Dimensionality reduction using t-SNE
    tsne = TSNE(n_components=2, perplexity=50)
    tsne.fit(X)
    X_train_pca = tsne.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_train_pca, y, test_size=0.2, random_state=42)
    # X_test_pca = tsne.transform(X_test)

else:
    raise ValueError("Invalid method selected.")

# # Train a k-means clustering model on the reduced data
# k = 10
# kmeans = KMeans(n_clusters=k)
# kmeans.fit(X_train, y_train)
# predicted_cluster = kmeans.predict(X_test)

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='none', random_state=0, max_iter=1000)
clf.fit(X_train, y_train)
predicted_cluster = clf.predict(X_test)

for i, value in enumerate(predicted_cluster):
    # Print the predicted cluster and actual label of the new record
    print("Predicted cluster:", value)
    print("Actual label:", y_test.iloc[i])

accuracy = accuracy_score(y_test, predicted_cluster)
print(accuracy)
