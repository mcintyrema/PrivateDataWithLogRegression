import pandas as pd
from sklearn.model_selection import train_test_split
import kAnonymizeData
import BiggerGroups
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# df = kAnonymizeData.df
# anon_result = kAnonymizeData.anon_result
df = BiggerGroups.df
anon_result = BiggerGroups.anon_result
anon_df = anon_result.dataset.to_dataframe()

# sampling
sample_df = anon_df.sample(frac=0.1, random_state=42)

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

# train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# get predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
