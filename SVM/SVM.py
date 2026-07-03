import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.datasets import make_blobs

X, y = make_blobs(
    n_samples=500,
    centers=2,
    n_features=2,
    cluster_std=3.2,
    random_state=42
)

y = np.where(y == 0, -1, 1)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1.0

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test  - mean) / std

X_vis = X[:, [0, 1]] 

w = np.zeros(X_train_scaled.shape[1])
b = 0

def SVM(X, y, w, b, C, lr=0.01, epochs=1000, lambda_param = 0.001):
    for epoch in range(epochs):
        for i in range(len(X)):
            loss = 0.5 * np.dot(w, w) + C*(max(0, 1 - y[i] * (np.dot(X[i], w) + b)))

            condition = y[i] * (np.dot(X[i], w) + b)    

            if condition >= 1:
                dw = 2 * lambda_param * w
                db = 0
            else:
                dw = 2 * lambda_param * w - y[i] * X[i]
                db = -y[i]

            w = w - lr * dw
            b = b - lr * db

    return w, b

def predict(X, w, b):
    score = np.dot(X, w) + b
    if score >= 0:
        return +1
    else:
        return -1
    
def accuracy(w, b):
    correct = 0
    total_samples = len(X_test_scaled)
    for i in range(total_samples):
        if predict(X_test_scaled[i], w, b) == y_test[i]:
            correct +=1
    return correct/total_samples * 100  


w, b = SVM(X_train_scaled, y_train, w, b, C=1, lr=0.001, epochs=1000)

margin_values = y_train * (np.dot(X_train_scaled, w) + b)
support_vector_indices = np.where(np.abs(margin_values - 1) <= 0.02)[0]
support_vectors = X_train_scaled[support_vector_indices]

x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
x_values = np.linspace(x_min, x_max, 100)

decision_boundary = -(w[0] * x_values + b) / w[1]

margin_positive = -(w[0] * x_values + b - 1) / w[1]
margin_negative = -(w[0] * x_values + b + 1) / w[1]

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

# ---------------- Original Dataset ----------------
ax[0].scatter(X[:, 0], X[:, 1], c=y, s=40)
ax[0].set_title("Original Synthetic Dataset")
ax[0].set_xlabel("Feature 1")
ax[0].set_ylabel("Feature 2")
ax[0].grid(True)

# ---------------- Decision Boundary ----------------
ax[1].scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, s=40)

ax[1].plot(x_values, decision_boundary,
           linewidth=2,
           label="Decision Boundary")

ax[1].plot(x_values, margin_positive,
           linestyle="--",
           linewidth=2,
           label="Positive Margin")

ax[1].plot(x_values, margin_negative,
           linestyle="--",
           linewidth=2,
           label="Negative Margin")

ax[1].scatter(
    support_vectors[:, 0],
    support_vectors[:, 1],
    s=180,
    facecolors="none",
    edgecolors="black",
    linewidths=2,
    label="Support Vectors"
)

ax[1].set_title("Linear SVM Decision Boundary")
ax[1].set_xlabel("Scaled Feature 1")
ax[1].set_ylabel("Scaled Feature 2")
ax[1].legend()
ax[1].grid(True)

plt.tight_layout()
plt.show()

print("Weights:", w)
print("Bias:", b)
print("Number of Support Vectors:", len(support_vectors))
print("Accuracy:", accuracy(w, b))
print("Norm of w:", np.linalg.norm(w))