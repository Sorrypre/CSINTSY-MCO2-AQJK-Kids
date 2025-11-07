# pandas for the processing of final_annotations.csv
import pandas as pd
# for array numerical computing used for multi-dimensional arrays
import numpy as np

# Regular expressions
import re

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import <insert preprocessing module for Multinomial Naive Bayes>
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# For affixing feature
import affixing as afx

def feature_is_capitalized(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) or not subj[0].isupper() else 1

def feature_vowel_count(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) else len(re.findall(r'[aeiou]', subj))
    
def feature_word_length(r):
    subj = r['word']
    return 0 if pd.isna(subj) else len(subj)

def feature_non_pure_abakada_count(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) else len(re.findall(r'[cfjqvxz]', subj))
    
def feature_fil_affix_sum(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) else afx.has_fil_affixing(subj)

def main():
    # Interpret
    unprocessed_data = pd.read_csv('final_annotations.csv', dtype={'word': str}, keep_default_na=False, na_values=[], na_filter=False)
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
    #print(unprocessed_data[['word', 'isFirstLetterCapital', 'numVowels', 'wordLength', 'numNonPureAbakada', 'filAffixSum']].to_string())
    # To check na values (naalala ko na yung code)
    na_rows = unprocessed_data['word'].isna()
    #print(unprocessed_data[na_rows])
    #print("Rows with empty word: ", na_rows.sum())

    X = unprocessed_data[['word', 'isFirstLetterCapital', 'numVowels', 'wordLength', 'numNonPureAbakada', 'filAffixSum']]
    y = unprocessed_data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, stratify=y)

    # column transfomer to for the model to better understand the features and drop irrelevant columns
    
    # character N-grams for the word column (2-4)
    word_transfomer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
    # numerical features scaling for good practice raw
    numerical_transfomer = Pipeline(steps=[('scaler', StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[('nGramText', word_transfomer, 'word'), 
                                                    ('scaledNum', numerical_transfomer, ['isFirstLetterCapital', 'numVowels', 'wordLength', 'numNonPureAbakada', 'filAffixSum'])], remainder='drop')
    
    final_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifer', LogisticRegression(solver='lbfgs', max_iter=2**15-1))]) # random_state=69

    # Training the model
    final_model.fit(X_train, y_train)

    # Evaluation of the model
    print(X_test)
    y_predict = final_model.predict(X_test)
    print(f"Prediction target test values: {y_predict}")
    print(f"Actual target test values:   {y_test}")
    acc_score = accuracy_score(y_test, y_predict)
    print(f"Model Accuracy on Test Set: {acc_score:.4f}")
    print(classification_report(y_test, y_predict))
    #print(confusion_matrix(y_test, y_predict))

    

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