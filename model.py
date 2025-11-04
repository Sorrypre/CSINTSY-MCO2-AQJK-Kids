# pandas for the processing of final_annotations.csv
import pandas as pd
# for array numerical computing used for multi-dimensional arrays
import numpy as np

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import <insert preprocessing module for Multinomial Naive Bayes>
from sklearn.naive_bayes import MultinomialNB
from  sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Pang-interpret ng .csv
import unpack_csv as unp

# For affixing feature
import affixing as afx

def feature_is_capitalized(r):
    subj = r['word']
    return False if pd.isna(subj) or not len(subj) else subj[0].isupper()

def feature_vowel_count(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) else subj.count(r'[aeiou]')
    
def feature_word_length(r):
    subj = r['word']
    return 0 if pd.isna(subj) else len(subj)

def feature_non_pure_abakada_count(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) else subj.count(r'[cfjqvxz]')
    
def feature_fil_affix_sum(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) else afx.has_fil_affixing(subj)

def main():
    # Interpret
    unprocessed_data = unp.csv_makepd('final_annotations.csv')
    # Feature 1
    unprocessed_data['isFirstLetterCapital'] = unprocessed_data.apply(feature_is_capitalized, axis=1)
    # Feature 2
    unprocessed_data['numVowels'] = unprocessed_data.apply(feature_vowel_count, axis=1)
    # Feature 3
    unprocessed_data['wordLength'] = unprocessed_data.apply(feature_word_length, axis=1)
    # Feature 4
    unprocessed_data['numNonPureAbakada'] = unprocessed_data.apply(feature_non_pure_abakada_count, axis=1)
    # Feature 5
    unprocessed_data['filAffixSum'] = unprocessed_data.apply(feature_fil_affix_sum, axis=1)
    
    # Test run
    print(unprocessed_data[['word', 'isFirstLetterCapital', 'numVowels', 'wordLength', 'numNonPureAbakada', 'filAffixSum']].to_string())

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