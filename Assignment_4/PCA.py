import numpy as np


class PCA:
    def __init__(self, new_dim: int) -> None:
        # hyperparameter representing the number of dimensions after reduction
        self.new_dim = new_dim
        # for standardization
        self.μ: np.ndarray
        self.σ: np.ndarray
        # for PCA
        self.A: np.ndarray

    # x_train is (m,n) matrix where each row is an n-dimensional vector of features
    def fit(self, x_train):
        # Find μ and σ of each feature in x_train
        M: int = x_train.shape[0]
        N: int = x_train.shape[1]
        self.μ = np.mean(x_train, axis=0)
        self.σ = np.std(x_train, axis=0, ddof=0)
        # if a column has zero std (useless constant) set σ=1 (skip their standardization)
        self.σ = np.where(self.σ == 0, 1, self.σ)

        #  Standardize the training data
        z_train = (x_train - self.μ) / self.σ

        # Compute the covariance matrix
        Σ = np.cov(z_train, rowvar=False, ddof=0)

        # Compute eigenvalues and eigenvectors using Numpy
        λs, U = np.linalg.eig(Σ)
        λs, U = (
            λs.real,
            U.real,
        )  # sometimes a zero imaginary part can appear due to approximations

        # Find the sequence of indices that sort λs in descending order
        λs_indices = np.argsort(λs)[::-1]
        # Use it to sort λs and U
        λs = λs[λs_indices]
        U = U[:, λs_indices]

        # Select the top L eigenvectors and set A accordingly
        L = self.new_dim
        λs = λs[: L + 1]
        U = U[:, :L]

        self.A = U.T

        return self

    # x_val is (m,n) matrix where each row is an n-dimensional vector of features
    def transform(self, x_val):
        z_val = (x_val - self.μ) / self.σ

        # Apply the transformation equation
        z_transformed = z_val @ self.A.T

        return z_transformed

    def inverse_transform(self, z_val):
        # TODO 8: Apply the inverse transformation equation (including destandardization)
        x_subdomain = z_val @ self.A

        x_transformed = x_subdomain * self.σ + self.μ
        return x_transformed

    def fit_transform(self, x_train):
        return self.fit(x_train).transform(x_train)
