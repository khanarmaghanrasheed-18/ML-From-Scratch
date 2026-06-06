import numpy as np
import math as m
import pandas as pd

df = pd.read_csv("DATASETS/iris.csv")
feature_cols = ["sepal.length", "sepal.width", "petal.length","petal.width"]
label_col = "variety"

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")


df = df.dropna(subset=feature_cols + [label_col])
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df[feature_cols].to_numpy().astype(float)
Y = df[label_col].to_numpy()

print(f"X shape: {X.shape}")
print(f"Y shape: {Y.shape}")
print(f"Classes: {set(Y)}")

# --- Train/test split ---
np.random.seed(42)
m_total = X.shape[0]
indices = np.random.permutation(m_total)
split_idx = int(0.7 * m_total)

X_train = X[indices[:split_idx]]
y_train = Y[indices[:split_idx]]
X_test  = X[indices[split_idx:]]
y_test  = Y[indices[split_idx:]]

# --- Setosaize using train stats only ---
mean = np.mean(X_train, axis=0)
std  = np.std(X_train, axis=0)
std[std == 0] = 1.0

X_train_scaled = (X_train - mean) / std
X_test_scaled  = (X_test  - mean) / std



def sort_array(arr):
    for i in range(len(arr)):
        for j in range(len(arr)-i-1):
            if arr[j][1] > arr[j+1][1]:
                temp = arr[j+1]
                arr[j+1] = arr[j]
                arr[j] = temp

def euclidean(a, b):
    total = 0.0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return m.sqrt(total)

def KNN(X, Y, x, k = 5):
    dist = []
    Setosa_count=0
    Versicolor_count=0
    Virginica_count = 0
    for i in range(len(X)):
        dist.append([i, euclidean(X[i], x)])
    sort_array(dist)
    for i in range(k):
        if Y[dist[i][0]] == "Setosa":
            Setosa_count+=1
        elif Y[dist[i][0]] == "Virginica":
            Virginica_count+=1
        elif Y[dist[i][0]] == "Versicolor":
            Versicolor_count+=1

    return Setosa_count, Versicolor_count, Virginica_count

def Weighted_KNN(X, Y, x, k = 5):
    dist = []
    Setosa_weight=0
    Versicolor_weight=0
    Virginica_weight = 0

    for i in range(len(X)):
        dist.append([i, euclidean(X[i], x)])
    sort_array(dist)

    for i in range(k):
        if dist[i][1] == 0:
            dist[i][1] = 0.00001
        if Y[dist[i][0]] == "Setosa":
            Setosa_weight+= 1/dist[i][1]
        elif Y[dist[i][0]] == "Virginica":
            Virginica_weight+= 1/dist[i][1]
        elif Y[dist[i][0]] == "Versicolor":
            Versicolor_weight+= 1/dist[i][1]

    return Setosa_weight, Versicolor_weight, Virginica_weight



def predict_KNN(Setosa, Versicolor, Virginica):
    if Setosa > Versicolor and Setosa > Virginica:
        return "Setosa"
    if Versicolor > Setosa and Versicolor > Virginica:
        return "Versicolor"
    if Virginica > Versicolor and Virginica > Setosa:
        return "Virginica"
    return "Tie"

# --- Accuracy over full test set ---
knn_correct  = 0
wknn_correct = 0

for i in range(len(X_test_scaled)):
    Setosa, Versicolor, Virginica = KNN(X_train_scaled, y_train, X_test_scaled[i])
    weighted_Setosa, weighted_Versicolor, weighted_Virginica = Weighted_KNN(X_train_scaled, y_train, X_test_scaled[i])
    knn_pred  = predict_KNN(Setosa, Versicolor, Virginica)
    wknn_pred = predict_KNN(weighted_Setosa, weighted_Versicolor, weighted_Virginica)
    if knn_pred  == y_test[i]:
        knn_correct  += 1
    if wknn_pred == y_test[i]:
        wknn_correct += 1

print(f"\n------Accuracy (k=5, test size={len(y_test)})------")
print(f"KNN Accuracy:          {knn_correct}/{len(y_test)} = {round(knn_correct/len(y_test)*100, 2)}%")
print(f"Weighted KNN Accuracy: {wknn_correct}/{len(y_test)} = {round(wknn_correct/len(y_test)*100, 2)}%")


print("\n------Predict Your Own Flower------")
print("Enter the 4 measurements to classify a new iris flower.")

sepal_length = float(input("Sepal Length (e.g. 5.1): "))
sepal_width  = float(input("Sepal Width  (e.g. 3.5): "))
petal_length = float(input("Petal Length (e.g. 1.4): "))
petal_width  = float(input("Petal Width  (e.g. 0.2): "))


user_point = np.array([sepal_length, sepal_width, petal_length, petal_width])
user_point_scaled = (user_point - mean) / std

Setosa, Versicolor, Virginica = KNN(X_train_scaled, y_train, user_point_scaled)
weighted_Setosa, weighted_Versicolor, weighted_Virginica = Weighted_KNN(X_train_scaled, y_train, user_point_scaled)

print("\n------KNN------")
print(f"Setosa Neighbours:    {Setosa}")
print(f"Versicolor Neighbours:{Versicolor}")
print(f"Virginica Neighbours: {Virginica}")
print(f"Prediction: {predict_KNN(Setosa, Versicolor, Virginica)}")

print("\n------Weighted KNN------")
print(f"Setosa weight:    {round(weighted_Setosa, 4)}")
print(f"Versicolor weight:{round(weighted_Versicolor, 4)}")
print(f"Virginica weight: {round(weighted_Virginica, 4)}")
print(f"Prediction: {predict_KNN(weighted_Setosa, weighted_Versicolor, weighted_Virginica)}")