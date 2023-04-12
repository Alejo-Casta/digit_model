import numpy as np


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.mean = None
        self.components = None

    def fit(self, X):
        # Compute mean of input data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Compute covariance matrix
        cov = np.cov(X, rowvar=False)

        # Compute eigenvectors and eigenvalues of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvectors by decreasing eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Store the first n_components eigenvectors as components
        self.components = eigenvectors[:, :self.n_components]

    def transform(self, X):
        # Project input data onto the first n_components eigenvectors
        X = X - self.mean
        return np.dot(X, self.components)

    def fit_transform(self, X):
        # Compute mean of input data
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # Compute covariance matrix
        cov = np.cov(X, rowvar=False)

        # Compute eigenvectors and eigenvalues of covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort eigenvectors by decreasing eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # Store the first n_components eigenvectors as components
        self.components = eigenvectors[:, :self.n_components]

        # Project input data onto the first n_components eigenvectors
        return np.dot(X, self.components)
