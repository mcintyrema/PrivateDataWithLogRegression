import pandas as pd
import diffprivlib.models as dp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from diffprivlib.utils import check_random_state

df = pd.read_csv("../adult.csv")

# sampling
sample_df = df.sample(frac=0.1, random_state=42)

# create numeric binary classifier
income_mapping = {
    ">50K": 1,
    "<=50K": 0,
    '*': -1
}
df['income'].str.strip()
sample_df['income'] = df['income'].map(income_mapping)

X = sample_df.drop(columns=['income'])
y = sample_df['income']  

# convert categorical to numeric
X = pd.get_dummies(X)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# epsilon values to try
epsilon_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50, 100.0, 500.0, 1000, 10000, float("inf")]
accuracies = []

for epsilon in epsilon_values:
    dp_clf = dp.LogisticRegression(epsilon=epsilon, data_norm=1e5)
    dp_clf.fit(X_train, y_train)
    
    accuracy = dp_clf.score(X_test, y_test)
    accuracies.append(accuracy)
    
    print(f"Epsilon: {epsilon}, Accuracy: {accuracy:.4f}")

# Plot accuracy vs epsilon
plt.figure(figsize=(10, 6))
plt.plot(epsilon_values, accuracies, marker='o')
plt.xscale('log')
plt.xlabel('Epsilon')
plt.ylabel('Accuracy')
plt.title('Accuracy vs Epsilon')
plt.grid(True)
plt.show()