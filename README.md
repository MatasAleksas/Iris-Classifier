Iris Flower Classifier:
-

This project uses machine learning to correctly identify iris flowers into 3 species. Setosa, Versicolor, and Virginica. Based on their unique features. The model is built using scikit-learn and trained on the iris data set.

Project Structure:
-

iris_classifier.py - Contains all steps from loading data to evaluation.

Features Used:
-
- Sepal length (cm)
- Sepal width (cm)
- Petal length (cm)
- Petal width (cm)

Tools and Libraries Used:
-
- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

Steps Performed:
-
1. Data set was loaded
2. Dataframe was created
3. Pair-plot was created and shown
4. Data was preprocessed
5. Split training and testing data
6. Creation of model and training it
7. Testing the model
8. Output of the results

How To Run:
-
1. Make sure you have Python installed
2. Install dependencies: `pip install pandas scikit-learn matplotlib seaborn`
3. Run the script: `python iris_classifier.py`

Notes:
-
- Expanded to also use KNN
- I made some of the bits of code into functions so we can call what we want
- Saves some time, like putting the pair plot into its own function