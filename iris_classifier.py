"""
Program that utilizes machine learning to classify different iris flowers.
Classify iris flowers into 3 species: Setosa, Versicolor, Virginica.

Algorithm Used: Logistic Regression, KNN.
Evaluation: Accuracy score, confusion matrix, visualization.

Author: Matas Aleksas
Version: 1.0.0
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Gets all the data
iris = load_iris()

# Puts in the flower info in the x, puts the flower name in the y
x = iris.data
y = iris.target

# Gets all the names of the features of the flowers in feature_names and the names of the flowers in target_names
feature_names = iris.feature_names
target_names = iris.target_names

# Initialize dataframe
df = pd.DataFrame(x, columns=feature_names)
df["species"] = pd.Categorical.from_codes(y, target_names)

# Creation and showing of a pair-plot of the Iris data
sns.pairplot(df, hue="species")
plt.title("Iris Data Pair-plot")
plt.show()

# Preprocessing the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Getting a random split of the training and testing data between the actual data and the names of the flowers
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Creating the models and training it
logistic = LogisticRegression()
logistic.fit(x_train, y_train)

knn = KNeighborsClassifier()
knn.fit(x_train, y_train)

# Testing the model with the testing data
log_pred = logistic.predict(x_test)
knn_pred = knn.predict(x_test)

# Outputting the results of the model
print("Logistic Regression Results: \n")
print("Accuracy: \n" + str(accuracy_score(y_test, log_pred)))
print("\nClassification Report: \n" + str(classification_report(y_test, log_pred)))
print("\nConfusion Matrix: \n" + str(confusion_matrix(y_test, log_pred)) + "\n")

print("KNN Results: \n")
print("Accuracy: \n" + str(accuracy_score(y_test, knn_pred)))
print("\nClassification Report: \n" + str(classification_report(y_test, knn_pred)))
print("\nConfusion Matrix: \n" + str(confusion_matrix(y_test, knn_pred)) + "\n")
