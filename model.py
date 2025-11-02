# pandas for the processing of final_annotations.csv
import pandas as pd
# for array numerical computing used for multi-dimensional arrays
import numpy as np

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import <insert preprocessing module for Multinomial Naive Bayes>
from sklearn.naive_bayes import MultinomialNB
from  sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def main():
    # Dataframe derived from the final_annotations.csv
    unprocessed_data = pd.read_csv("final_annotations.csv")
    print(unprocessed_data)
    unprocessed_data.info()
    print(unprocessed_data.isnull().sum())
    essential_data = drop_irrelevant_columns(unprocessed_data)
    print(essential_data)


# Data cleaning: Removing irrelevant columns for feature matrix
# Droped is_ne and is correct spelling for now kasi sabi ni sir pwede gamitin although not necessary 
def drop_irrelevant_columns(df):
    df_essential = df.drop(columns=["word_id", "sentence_id", "is_ne", "is_spelling_correct"])
    return df_essential

# feature engineering
# 1. prefix/suffix/Character-level n-grams, word Lengths(numerical), capitilization(True or false)
# 2. Transformation: for prefix/suffix/Character-level n-grams apply one hot encoding, others as is
# 3. Use sklearn.compose.ColumnTransformer or sklearn.feature_selection.FeatureUnion to combine the
#    outputs of the different feature extraction intu a unified feature matrix for the classifier
# insert additional features logic here

# handle missing values 

# Train test split

# Model training
    
if __name__ == "__main__":
    main()