import numpy as np
import pandas as pd

# 1. Load Data
df = pd.read_csv("DATASETS/matches.csv")
feature_cols = ["runs_left", "balls_left", "wickets","req_run_rate"]

# 2. Force Numeric & Clean (The "Safety First" Block)
for col in feature_cols + ["result"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop any row that has a NaN in our features or the result
df = df.dropna(subset=feature_cols + ["result"])

# Remove any rows where run rates might be Infinite (e.g., division by zero in the CSV)
df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

X = df[feature_cols].to_numpy().astype(float)
Y = df["result"].to_numpy().astype(float)

print(f"X shape: {X.shape}")
print(f"y shape: {Y.shape}")

# 3. Train-Test Split
np.random.seed(42)
m_total = X.shape[0]
print(f"m_total: {X.shape[0]}")
indices = np.random.permutation(m_total)
split_idx = int(0.7 * m_total)

X_train = X[indices[:split_idx]]
y_train = Y[indices[:split_idx]]

X_test = X[indices[split_idx:]]
y_test = Y[indices[split_idx:]]

# 4. Scaling with Pre-calculation Check
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)

# If std is 0, it means the column is constant. We set it to 1 to avoid div by zero.
std[std == 0] = 1.0 

X_train_scaled = (X_train - mean) / std
X_test_scaled = (X_test - mean) / std

# 5. Model Setup
ones = np.ones((X_train_scaled.shape[0], 1))  # create vertical column of 1s (bias)
X_final = np.hstack((ones, X_train_scaled)) # glues to the front of the data to achieve y = w0(1) + wx1....
theta = np.zeros(X_final.shape[1]) # initialises weights with zeros according to no of columns

ones_test = np.ones((X_test_scaled.shape[0], 1))
X_test_final = np.hstack((ones_test, X_test_scaled))

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# 6. Gradient Descent
alpha = 0.5 # Increased learning rate for faster convergence
iterations = 3000
m = len(y_train)

for i in range(iterations):
    z = np.dot(X_final, theta)
    h = sigmoid(z)
    
    gradient = np.dot(X_final.T, (h - y_train)) / m # real calculus behind 
    theta = theta - alpha * gradient 
    
    if i % 100 == 0:
        # Prevent log(0) which causes nan
        h_stable = np.clip(h, 1e-15, 1 - 1e-15)
        # Likelihood maximization <-> Cost minimization
        cost = -np.mean(y_train * np.log(h_stable) + (1 - y_train) * np.log(1 - h_stable))
        print(f"Iteration {i}: Cost {cost}")

# 7. Prediction
print("\n--- Predict Win Probability ---")
try:
    runs_rem = float(input("Enter the runs remaining: "))
    balls_rem = float(input("Enter the balls remaining: "))
    wickets_rem = float(input("Enter the wickets remaining: "))
    rrr = float((runs_rem*6)/balls_rem)
    print(f"Required Run Rate: {rrr}")

    test_case = np.array([runs_rem, balls_rem, wickets_rem, rrr])
    test_scaled = (test_case - mean) / std
    test_final = np.insert(test_scaled, 0, 1.0) # adding bias terms

    prob = sigmoid(np.dot(test_final, theta))
    print(f"\nWin Probability: {prob * 100:.2f}%")
except ValueError:
    print("Please enter valid numbers.")

# 8. Final Accuracy Check
train_preds = (sigmoid(np.dot(X_final, theta)) >= 0.5).astype(int)
print(f"Final Training Accuracy: {np.mean(train_preds == y_train) * 100:.2f}%")

# --- Test Accuracy ---
test_probs = sigmoid(np.dot(X_test_final, theta))
test_preds = (test_probs >= 0.5).astype(int)

test_accuracy = np.mean(test_preds == y_test)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\n--- Model Weights ---")
print(f"Bias (b): {theta[0]:.4f}")
print(f"runs_left weight: {theta[1]:.4f}")
print(f"balls_left weight: {theta[2]:.4f}")
print(f"wickets weight: {theta[3]:.4f}")
print(f"req_run_rate weight: {theta[4]:.4f}")

# If you want next, I can show you something very powerful:

# 👉 how to plot cost vs iterations and detect:

# underfitting

# overfitting

# bad learning rate

# That’s where ML starts feeling like a real system instead of just math.