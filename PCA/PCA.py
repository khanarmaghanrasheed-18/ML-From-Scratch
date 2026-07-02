import numpy as np
import matplotlib.pyplot as plt

X = np.array([
    [150, 48],
    [155, 52],
    [160, 56],
    [165, 60],
    [170, 65],
    [175, 70],
    [180, 74],
    [185, 79],
    [190, 84]
], dtype=float)

def mean(X, d):
    sum = 0
    for i in range(len(X)):
        sum += X[i][d]        
    return sum/len(X)

def covariance(X, d1, d2):
    sum = 0
    for i in range(len(X)):
        sum += (X[i][d1] - mean(X, d1)) * (X[i][d2] - mean(X, d2))

    return sum

def variance(X, d):
    var = 0
    for i in range(len(X)):
        var += (X[i][d] - mean(X, d))**2
    return var


def covariance_matrix(X, dimensions):
    cov_X = [[0 for _ in range(dimensions)] for _ in range(dimensions)]
    for i in range(dimensions):
        for j in range(dimensions):
            cov_X[i][j] = covariance(X, i, j)

            if i == j:
                cov_X[i][j] =  variance(X, j)

    eigenvalues, eigenvectors = np.linalg.eig(cov_X)
    return eigenvalues, eigenvectors

def pca(X, dimensions, final_dimensions):
    X_centered = X - np.mean(X, axis=0)

    eigenvalues, eigenvectors = covariance_matrix(X_centered, dimensions)
    sorted_indices = np.argsort(eigenvalues)[::-1]

    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    W = eigenvectors[:, :final_dimensions]
    X_projected = np.dot(X_centered, W)

    return X_projected, W, eigenvalues, X_centered 

    
X_projected, W, eigenvalues, X_centered = pca(X, dimensions=2, final_dimensions=1)



fig, ax = plt.subplots(1, 3, figsize=(15, 5))


ax[0].scatter(X[:, 0], X[:, 1], s=80)
ax[0].set_title("Original Dataset")
ax[0].set_xlabel("Height (cm)")
ax[0].set_ylabel("Weight (kg)")
ax[0].grid(True)

ax[1].scatter(X_centered[:, 0], X_centered[:, 1], s=80)
ax[1].axhline(0, linestyle="--")
ax[1].axvline(0, linestyle="--")
ax[1].set_title("Mean-Centered Data")
ax[1].set_xlabel("Centered Height")
ax[1].set_ylabel("Centered Weight")
ax[1].grid(True)


ax[2].scatter(X_projected, np.zeros(len(X_projected)), s=80)
ax[2].set_title("Projection onto PC1")
ax[2].set_xlabel("Principal Component 1")
ax[2].set_yticks([])
ax[2].grid(True)

plt.tight_layout()
plt.show()

print("Original Shape:", X.shape)
print("Projected Shape:", X_projected.shape)
print("\nEigenvalues:")
print(eigenvalues)
print("\nPrincipal Component:")
print(W)



