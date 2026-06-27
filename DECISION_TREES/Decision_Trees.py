import numpy as np
import pandas as pd
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

m_total = X.shape[0]
indices = np.random.permutation(m_total)

split_idx = int(0.7 * m_total)

X_train = X[indices[:split_idx]]
y_train = Y[indices[:split_idx]]

X_test = X[indices[split_idx:]]
y_test = Y[indices[split_idx:]]

print("\n" + "="*50)
print("DECISION TREE CLASSIFIER FROM SCRATCH")
print("="*50)

print(f"\nDataset Size: {len(X)}")
print(f"Features Used: {feature_cols}")
print(f"Classes: {np.unique(Y)}")

print(f"\nTraining Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")

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

# def print_tree(node, depth=0):
#     indent = "    " * depth
#     if node.label is not None:
#         print(f"{indent}Leaf -> {node.label}")
#         return

#     feature_name = feature_cols[node.feature]
#     print(f"{indent}{feature_name} <= {node.threshold}")
#     print(f"{indent}├── True:")
#     print_tree(node.left, depth + 1)
#     time.sleep(0.7)

#     print(f"{indent}└── False:")
#     print_tree(node.right, depth + 1)


    
# def accuracy(y_true, y_pred):
#     correct = 0
#     for i in range(len(y_true)):
#         if y_true[i] == y_pred[i]:
#             correct += 1

#     return correct / len(y_true)

# def show_test_predictions(root, X_test, y_test):
#     print("\n--- TEST PREDICTIONS ---\n")

#     correct = 0
#     for i in range(len(X_test)):
#         pred = predict(root, X_test[i])
#         actual = y_test[i]
#         if i < 10:
#             print(f"Sample {i+1}:")
#             print(f"Features: {X_test[i]}")
#             print(f"Predicted: {pred} | Actual: {actual}")
#             time.sleep(0.4)
#             print("-" * 40)

#         if pred == actual:
#             correct += 1


#     accuracy = correct / len(X_test) * 100
#     print("\nAccuracy:", accuracy)


# print("\nBUILDING DECISION TREE...\n")
# time.sleep(1)
# print("TREE STRUCTURE:\n")
# print_tree(root)
# time.sleep(2)

# print("\nTESTING ON UNSEEN DATA...\n")
# time.sleep(2)
# show_test_predictions(root, X_test, y_test)

root = build_tree(X_train, y_train)
for i in range(2):
    print("\nNOW TRY A CUSTOM WINE SAMPLE\n")
    sample = []
    for name in feature_cols:
        val = float(input(f"Enter {name}: "))
        sample.append(val)

    sample = np.array(sample)
    result = predict(root, sample)
    print("\nPrediction:", result)