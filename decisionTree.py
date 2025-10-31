import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split

def main():
    iris = load_iris()
    
    # feature / the values associated with a certain feature
    x = np.array(iris['data'])

    # labels / target values / correct answers used for training
    y = np.array(iris['target'])

    numTargetData = len(y)
    numRowsFeatureMatrix = x.shape[0]

    print(f"Number of target data: {numTargetData}")
    print(f"Number of rows of feature matrix: {numRowsFeatureMatrix}")

    if (numTargetData != numRowsFeatureMatrix):
        print("Warning! The number of rows in the feature matrix does not match the number of target values\n")
    else:
        print("Matching number of rows in feature matrix and target values\n")

    ##########################
    #    TRAIN TEST SPLIT    #
    ##########################
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)


    ##########################
    #     MODEL TRAINING     #
    ##########################

    # Step 1: Initializing the model
    model = DecisionTreeClassifier()

    # Mold the data to train the model
    model.fit(x_train, y_train)

    # for visualizing the data
    # plt.figure(figsize = (12, 8))
    # plot_tree(model, feature_names=iris['feature_names'], class_names=iris['target_names'], filled=True, rounded=True)
    # plt.show()

    # Only the x_test should be put kasi yung y yung sagot eh ayun nga pinepredict
    predictions = model.predict(x_test)
    print(f"Prection target test values: {predictions}")
    # Now you can use y_test to check our model's performance
    print(f"Actual target test values:   {y_test}")
    # In SKLearn, you also have the confusion matrix, accuracy, percision, recall, etc.

    # In MCO2, we are the one who will decide about the feature
    # Not allowed to use dictionaries
    # not allowed the other existing language identification models

if __name__ == "__main__":
    main()