import pandas as pd
import numpy as np
import random
import time

np.set_printoptions(suppress=True)

columns = [
    "class","alcohol","malic_acid","ash","alcalinity_of_ash","magnesium",
    "total_phenols","flavanoids","nonflavanoid_phenols","proanthocyanins","color_intensity",
    "hue","od280_od315","proline"]

df = pd.read_csv("DATASETS/wine.data", header=None, names=columns)

feature_cols = ["alcohol", "flavanoids", "od280_od315", "color_intensity", "proline"]

X = df[feature_cols].to_numpy().astype(float)
Y = df["class"].to_numpy()

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=feature_cols + ["class"])
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

df = df.sample(frac=1, random_state=42).reset_index(drop=True)

np.random.seed(42)

def gini(labels):
    total = len(labels)
    if total == 0:
        return 0

    classes , counts = np.unique(labels, return_counts=True)
    impurity = 1

    for count in counts:
        p = count / total
        impurity -= p ** 2

    return impurity

def compute_impurity(X, y, threshold, feature):
    left_y = []
    right_y = []
    for i in range(len(X)):
        if X[i][feature] < threshold:
            left_y.append(y[i])
        else:
            right_y.append(y[i])

    left_count = len(left_y)
    right_count = len(right_y)
    total = left_count + right_count

    weighted_gini = ((left_count / total) * gini(left_y) + (right_count / total) * gini(right_y))
    return weighted_gini

def decision_tree(X, y):
    best_gini = 1
    best_threshold = None
    best_feature = None

    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])

        for i in range(len(values)-1):
            threshold = (values[i] + values[i+1]) / 2
            gini_score = compute_impurity(X, y, threshold, feature)
            if gini_score < best_gini:
                best_gini = gini_score
                best_threshold = threshold
                best_feature = feature

    return best_gini, best_threshold, best_feature

def split(X, y, threshold, feature):
    left_X = [] 
    right_X = []
    left_y = []
    right_y = []
    for i in range(len(X)):
        if X[i][feature]<threshold:
            left_X.append(X[i])
            left_y.append(y[i])
        else: 
            right_X.append(X[i])
            right_y.append(y[i])

    return left_X, right_X, left_y, right_y

class Node:
    def __init__(self, t= None, f = None, l = None, lc = None, rc = None):
        self.threshold = t
        self. feature = f
        self.label = l
        self.left = lc
        self.right = rc

def build_tree(X, y):
    if len(np.unique(y)) == 1:
        return Node(l=y[0])
    
    gini, threshold, feature = decision_tree(X, y)
    leftX, rightX, lefty, righty = split(X, y, threshold, feature)
    leftX = np.array(leftX)
    rightX = np.array(rightX)
    left_node = build_tree(leftX, lefty)
    right_node = build_tree(rightX, righty)
    return Node(threshold, feature, lc= left_node, rc= right_node)


def predict(node, sample):
    if node.label is not None:
        return node.label

    if sample[node.feature] < node.threshold:
        return predict(node.left, sample)
    else:
        return predict(node.right, sample)
 

def random_forest(X, y, K= 100):
    forest = []
    oob_votes = [[] for _ in range(len(X))]
    for i in range(K):
        bootstrap_indices = []

        for i in range(len(X)):
            random_number = random.randint(0, len(X)-1)
            bootstrap_indices.append(random_number)

        bootstrap_indices = np.array(bootstrap_indices)

        bootstrapped_X = X[bootstrap_indices]
        bootstrapped_Y = y[bootstrap_indices]

        OOB_indices = []

        for i in range(len(bootstrap_indices)):
            if i not in bootstrap_indices:
                OOB_indices.append(i)

        
        tree = build_tree(bootstrapped_X, bootstrapped_Y)
        forest.append(tree)

        for i in range(len(OOB_indices)):
            pred = predict(tree, X[OOB_indices[i]])
            oob_votes[OOB_indices[i]].append(pred)

    return forest, oob_votes

def predict_forest(forest, sample):
    votes = []

    for tree in forest:
        pred = predict(tree, sample)
        votes.append(pred)
    
    class1 = 0
    class2 = 0
    class3 = 0
    for vote in votes:
        if vote == 1:
            class1+=1
        elif vote==2:
            class2+=1
        elif vote==3:
            class3+=1

    if class1>class2 and class1>class3:
        return 1
    elif class2>class3 and class2>class1:
        return 2
    elif class3>class1 and class3>class2:
        return 3

def forest_accuracy_report(oob_votes, y):
    correct = 0
    total = 0

    print("\n" + "=" * 55)
    print("        RANDOM FOREST REPORT")
    print("=" * 55)

    for i in range(len(oob_votes)):

        if len(oob_votes[i]) == 0:
            continue

        count1 = 0
        count2 = 0
        count3 = 0

        for vote in oob_votes[i]:
            if vote == 1:
                count1 += 1
            elif vote == 2:
                count2 += 1
            elif vote == 3:
                count3 += 1

        if count1 >= count2 and count1 >= count3:
            prediction = 1
        elif count2 >= count1 and count2 >= count3:
            prediction = 2
        else:
            prediction = 3

        if prediction == y[i]:
            correct += 1

        total += 1

    accuracy = (correct / total) * 100

    print(f"\nTotal OOB samples checked : {total}")
    print(f"Correct predictions       : {correct}")
    print(f"Wrong predictions         : {total - correct}")
    print(f"Accuracy                  : {accuracy:.2f}%")

    print("\n" + "=" * 55)
    
forest, oob_votes = random_forest(X, Y, K=100)

print("\n--- ENTER VALUES FOR RANDOM FOREST PREDICTION ---\n")
sample = []
for name in feature_cols:
    val = float(input(f"Enter {name}: "))
    sample.append(val)

sample = np.array(sample)
result = predict_forest(forest, sample)
print("\nRandom Forest Prediction: Class ", result)

time.sleep(3)
forest_accuracy_report(oob_votes, Y)
