# Step 1: Importing necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import tree
import matplotlib.pyplot as plt

# Step 2: Load the dataset
data = pd.read_csv('german_credit_data.csv')

# Step 3: Data Preprocessing
# Checking for missing values
print(data.isnull().sum())

# Handling categorical variables (if present)
data = pd.get_dummies(data, drop_first=True)

# Step 4: Splitting the dataset into features and target
# Assuming the last column is the target variable
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Step 5: Split into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Model Building using DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='gini', random_state=42)

# Step 7: Train the model
classifier.fit(X_train, y_train)

# Step 8: Prediction on the test set
y_pred = classifier.predict(X_test)

# Step 9: Evaluation
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 10: Visualization of the decision tree
plt.figure(figsize=(20,10))
tree.plot_tree(classifier, filled=True, feature_names=X.columns, class_names=['No', 'Yes'], rounded=True)
plt.show()
