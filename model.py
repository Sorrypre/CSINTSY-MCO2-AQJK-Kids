# pandas for the processing of final_annotations.csv
import pandas as pd
# for array numerical computing used for multi-dimensional arrays
import numpy as np

# Regular expressions
import re

from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import <insert preprocessing module for Multinomial Naive Bayes>
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# For affixing feature
import affixing as afx

# Grammar and morphology assessment
import gma

# For turning list into a passable parameter for model.predict
import word_parser as wpar

# For dynamic assignment of features in word_parser
feature_columns = [
    'word',
    'isFirstLetterCapital',
    'numVowels',
    'wordLength',
    'numNonPureAbakada',
    'filAffixSum',
    'filPronoun',
    'engPronoun',
    #'filPrefix',
    #'filInfix',
    #'filSuffix',
    'filFirstCluster',
    'filMiddleCluster',
    'engFirstCluster',
    'engMiddleCluster',
    'engEndCluster',
    'ne_tag',
]

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
    return 0 if pd.isna(subj) or not len(subj) else len(re.findall(r'[fqvxz]', subj))
    
def feature_fil_affix_sum(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) else afx.has_fil_affixing(subj)
            
def feature_fil_pronoun(r):
    subj = r['word']
    return '' if pd.isna(subj) or not len(subj) else gma.return_or_empty_if_fil(subj)

def feature_eng_pronoun(r):
    subj = r['word']
    return '' if pd.isna(subj) or not len(subj) else gma.return_or_empty_if_eng(subj)

def feature_ffc(r):
    subj = r['word']
    return '' if pd.isna(subj) or not len(subj) else gma.fil_first_cluster(subj)

def feature_fmc(r):
    subj = r['word']
    return '' if pd.isna(subj) or not len(subj) else gma.fil_middle_cluster(subj)
    
def feature_efc(r):
    subj = r['word']
    return '' if pd.isna(subj) or not len(subj) else gma.eng_first_cluster(subj)

def feature_emc(r):
    subj = r['word']
    return '' if pd.isna(subj) or not len(subj) else gma.eng_middle_cluster(subj)

def feature_eec(r):
    subj = r['word']
    return '' if pd.isna(subj) or not len(subj) else gma.eng_end_cluster(subj)

def feature_ne_tag(r):
    subj = r['is_ne']
    return 0 if pd.isna(subj) or not len(subj) else 1
    
def get_feature(n):
    if n == 1:
        return feature_is_capitalized
    elif n == 2:
        return feature_vowel_count
    elif n == 3:
        return feature_word_length
    elif n == 4:
        return feature_non_pure_abakada_count
    elif n == 5:
        return feature_fil_affix_sum
    elif n == 6:
        return feature_fil_pronoun
    elif n == 7:
        return feature_eng_pronoun
    elif n == 8:
        return feature_ffc
    elif n == 9:
        return feature_fmc
    elif n == 10:
        return feature_efc
    elif n == 11:
        return feature_emc
    elif n == 12:
        return feature_eec
    elif n == 13:
        return feature_ne_tag
    else:
        raise Exception('feature out of bounds')

def custom_model_test(model, prompt, expectations):
    if not len(prompt) or not len(expectations):
        raise Exception('prompt or expectation is empty')
    if len(prompt) != len(expectations):
        raise Exception('tokens in prompt and tokens in expectations should be equal')
    predictions = model.predict(wpar.pdfy(prompt))
    # Count corrects and incorrects
    correct = 0
    incorrects = []
    for i in range(len(prompt)):
        if predictions[i] == expectations[i]:
            correct += 1
        else:
            incorrects.append([i, prompt[i], f"exp={expectations[i]}", f"pred={predictions[i]}"])
    custom_accuracy = correct / len(prompt) * 100.0
    print(f"Prompt: {prompt}")
    print(f"Expected: {expectations}")
    print(f"Custom prediction results: {predictions}")
    print(f"Prediction accuracy: {custom_accuracy:.2f}%")
    if len(incorrects):
        print("Incorrects:")
        for i in incorrects:
            print(i)

def main():
    # Interpret
    ann0 = pd.read_csv('final_annotations.csv', dtype={'word': str}, keep_default_na=False, na_values=[], na_filter=False)
    ann1 = pd.read_csv('ann1.csv', dtype={'word': str, 'label': str, 'is_ne': str}, keep_default_na=False, na_values=[], na_filter=False)
    # Drop irrelevant columns
    ann0 = drop_irrelevant_columns(ann0)
    # Copy to another column if it is a pronoun
    ann_full = pd.concat([ann0, ann1], ignore_index=True, sort=False)
    # Get adjacent words
    ann_full['previous_word'] = ann_full['word'].shift(1, fill_value='')
    ann_full['next_word'] = ann_full['word'].shift(-1, fill_value='')
    # Make a column for each feature
    for f in range(1, len(feature_columns)):
        ann_full[feature_columns[f]] = ann_full.apply(get_feature(f), axis=1)
    #print(ann_full)
    
    X = ann_full[feature_columns + ['previous_word', 'next_word', 'is_ne']]
    y = ann_full['label']
    
    # Split for a train-test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=0)
    # Split again for a validation set
    X_train2, X_validation, y_train2, y_validation = train_test_split(X_train, y_train, test_size=0.50, stratify=y_train, random_state=0)
    
    word_transformer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
    comma_tokenizer = lambda s: [t.strip() for t in s.split(',')]
    tag_transformer = CountVectorizer(binary=True)
    numerical_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])
    preprocessor = ColumnTransformer(transformers=[
        ('text', word_transformer, 'word'),
        ('numeric', numerical_transformer, feature_columns[1:6] + [feature_columns[13]]),
        ('cf_pr', categorical_transformer, [feature_columns[6]]),
        ('ce_pr', categorical_transformer, [feature_columns[7]]),
        ('cf_fc', tag_transformer, feature_columns[8]),
        ('cf_mc', tag_transformer, feature_columns[9]),
        ('ce_fc', tag_transformer, feature_columns[10]),
        ('ce_mc', tag_transformer, feature_columns[11]),
        ('ce_ec', tag_transformer, feature_columns[12])
    ], remainder='drop')
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(solver='saga', max_iter=2**15-1))
    ])
    
    # Train model
    model.fit(X_train2, y_train2)
    
    # Interpret results
    y_predict = model.predict(X_test)
    print(f"Prediction target test values: {y_predict}")
    print(f"Actual target test values:   {y_test}")
    acc_score = accuracy_score(y_test, y_predict)
    print(f"Model accuracy on test set: {acc_score:.4f}")
    print(classification_report(y_test, y_predict))
    
    # Custom prediction
    prompt = ['pain', 'in', 'the', 'ass']
    expectations = ['ENG', 'ENG', 'ENG', 'ENG']
    custom_model_test(model, prompt, expectations)

# Data cleaning: Removing irrelevant columns for feature matrix
# Droped is_ne and is correct spelling for now kasi sabi ni sir pwede gamitin although not necessary 
def drop_irrelevant_columns(df):
    df_essential = df.drop(columns=["word_id", "sentence_id", "is_spelling_correct"])
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