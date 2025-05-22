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


# Preprocessing the data
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Getting a random split of the training and testing data between the actual data and the names of the flowers
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

"""
Pair plot to display iris data.
Takes too long to run so I put it in a separate function so the user has the option of displaying it.
"""
def plot():
    # Creation and showing of a pair-plot of the Iris data
    sns.pairplot(df, hue="species")
    plt.title("Iris Data Pair-plot")
    plt.show()

"""
Predicts the outcome of given inputs as a 0 or 1.
Given the data, is it one of the specific iris types.
"""
def logistic_regression():
    # Creating and training the logistic regression model
    # Testing the model
    # Outputting the results

    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)

    log_pred = logistic.predict(x_test)

    print("Logistic Regression Results: \n")
    print("Accuracy: \n" + str(accuracy_score(y_test, log_pred)))
    print("\nClassification Report: \n" + str(classification_report(y_test, log_pred)))
    print("\nConfusion Matrix: \n" + str(confusion_matrix(y_test, log_pred)) + "\n")

"""
Works by finding the closest data point to a new given point.
Given a point what iris type is it closest too given its data. 
"""
def knn_model():
    # Creation and training
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)

    # Testing
    knn_pred = knn.predict(x_test)

    # Results
    print("KNN Results: \n")
    print("Accuracy: \n" + str(accuracy_score(y_test, knn_pred)))
    print("\nClassification Report: \n" + str(classification_report(y_test, knn_pred)))
    print("\nConfusion Matrix: \n" + str(confusion_matrix(y_test, knn_pred)) + "\n")

"""
Main function. All functions called inside of here.
"""
if __name__ == '__main__':
    # Checks used to see if the functions were already called, we dont want to run the same model twice
    # I'm afraid it'll break something if it does.
    log_used = False
    knn_used = False

    # Asks user if they want to see the pair plot
    # Program runs slightly faster if we have this so it's a win
    while True:
        print("Would you like to display a pair plot of the iris data?")
        map_choice = input("0 or 1 for no or yes: ")

        if map_choice == "1":
            print("Displaying pair plot...\n")
            plot()
            break
        elif map_choice == "0":
            print("Continuing program...\n")
            break
        else:
            print("Invalid input")

    # Main block of code that asks the user for input
    while True:
        # If all the options were exhausted then it exits the program
        if log_used and knn_used:
            print("All models have been used. Exiting program.")
            break

        print("Iris Classifier Model:")
        print("Pick an option (0-2) for what model you want to use.")
        print("Each model can only be used once.\n")
        print("0. Logistic Regression")
        print("1. KNN")
        print("2. Exit")

        choice = input()

        # Checks to see what the choices are
        if choice == "0":
            if log_used:
                print("Logistic Regression model already used. Pick another option.")
                continue
            else:
                logistic_regression()
                log_used = True
        elif choice == "1":
            if knn_used:
                print("KNN model already used. Pick another option.")
                continue
            else:
                knn_model()
                knn_used = True
        elif choice == "2":
            print("Exiting...")
            break
        else:
            print("Invalid option. Please try again.")