import pandas as pd
import numpy as np
import random 
import math as m
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

df = pd.read_excel("DATASETS/Mall_Customers.xlsx")

feature_cols = ["Annual Income (k$)", "Spending Score (1-100)"]
for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=feature_cols)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X_original = df[feature_cols].to_numpy().astype(float)

mean = np.mean(X_original, axis=0)
std = np.std(X_original, axis=0)
std[std == 0] = 1

X = (X_original - mean) / std
np.random.seed(42)

plt.figure(figsize=(7,6))

plt.scatter(
    X_original[:,0],
    X_original[:,1],
    color='steelblue',
    edgecolors='black'
)

plt.title("Mall Customers Dataset (Before Clustering)")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.grid(True)

plt.show()

def euclidean(a, b):
    total = 0.0
    for i in range(len(a)):
        total += (a[i] - b[i]) ** 2
    return m.sqrt(total)

def are_centroids_changing(X, cluster, K = 3):
    new_centroids = []

    for i in range(K):
        length = len(cluster[i])
        if length == 0:
            new_centroids.append(X[random.randint(0, len(X)-1)])
            continue

        sumX = 0
        sumY = 0
        for j in range(length):
            sumX += cluster[i][j][0]
            sumY += cluster[i][j][1]
        average = (sumX/length, sumY/length)
        new_centroids.append(average)

    return new_centroids


def evaluate_dist(X, centroids, K = 3):
    cluster = [[] for i in range(K)]
    for i in range(len(X)):
        distances = []
        for j in range(len(centroids)):
            distances.append(euclidean(X[i], centroids[j]))
        nearest = distances.index(min(distances))
        cluster[nearest].append(X[i])

    new_centroids = are_centroids_changing(X, cluster, K)

    for i in range(K):
        if euclidean(centroids[i], new_centroids[i]) > 0.001:
            return evaluate_dist(X, new_centroids, K)
        else :
            continue
    
    return new_centroids, cluster       
    

def k_means(X, K = 3):
    centroids = []
    for i in range(3):
        centroids.append(X[random.randint(0, len(X)-1)])

    updated_centroids, clusters = evaluate_dist(X, centroids, K)

    return updated_centroids, clusters

K = 5

updated_centroids, clusters = k_means(X, K)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'black', 'cyan']

for i in range(len(clusters)):

    cluster = np.array(clusters[i])

    plt.scatter(
        cluster[:,0],
        cluster[:,1],
        color=colors[i],
        label=f'Cluster {i+1}'
    )

centroids = np.array(updated_centroids)

plt.scatter(
    centroids[:,0],
    centroids[:,1],
    marker='X',
    s=250,
    color='yellow',
    edgecolors='black',
    label='Centroids'
)

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title(f"K-Means Clustering with K = {K}")
plt.legend()
plt.grid(True)

plt.show()

print(f"{'Cluster':<10}{'Size':<10}{'Centroid'}")

for i in range(len(clusters)):
    print(f"{i+1:<10}{len(clusters[i]):<10}{np.round(updated_centroids[i],3)}")