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
from sklearn.preprocessing import StandardScaler, QuantileTransformer

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from  sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report

# For affixing feature
import affixing as afx

import word_parser as wpar

# For dynamic assignment of features in word_parser
feature_columns = [
    'word',
    'isFirstLetterCapital',
    'numVowels',
    'wordLength',
    'numNonPureAbakada',
    'filAffixSum',
    'pronoun_type',
    'ne_tag',
]

fil_pronouns = [
    'ikaw', 'kaw', 'ka', 'mo', 'iyo', 'sayo',
    'ako', 'ko', 'akin', 'sakin',
    'siya', 'sya', 'niya', 'kaniya', 'sakaniya',
    'sila', 'nila', 'kanila', 'sakanila',
    'kayo', 'kamo', 'ninyo', 'niyo', 'inyo', 'sainyo',
    'tayo', 'natin', 'atin', 'satin',
    'kami', 'namin', 'amin', 'samin',
    
    'ito', 'eto', 'heto', 'nito', 'neto', 'dito', 'nandito', 'narito', 'ganito', 'ganto',
    'iyan', 'yan', 'hayan', 'niyan', 'nyan', 'diyan', 'dyan', 'jan', 'nandiyan', 'nandyan', 'nanjan', 'ganiyan', 'ganyan', 'gay-an', 'gayan',
    'iyon', 'iyun', 'yon', 'yun', 'yaon', 'hayon', 'niyon', 'noon', 'nun', 'doon', 'dun', 'nandoon', 'nandun', 'naroon', 'naron', 'ayon', 'ayun',
    
    'ano', 'sino', 'saan', 'san', 'alin', 'gaano', 'gano', 'ilan', 'bakit', 'paano', 'pano', 'kailan', 'kelan',
    'anuman', 'sinoman', 'sinuman', 'saanman', 'sanman', 'alinman', 'gaanoman', 'ganoman', 'ganuman', 'ilanman', 'paanoman', 'panoman', 'panuman', 'kailanman', 'kelanman'
]

eng_pronouns = [
    'I', 'i', 'me', 'myself', 'mine', 'my',
    'we', 'us', 'ourself', 'ourselves', 'ours', 'our',
    'you', 'u', 'yourself', 'urself', 'yourselves', 'urselves', 'yours', 'urs', 'your', 'ur',
    'thou', 'thee', 'thyself', 'theeself', 'thine',
    'yall', 'yallselves',
    'he', 'him', 'himself', 'hisself', 'his',
    'she', 'her', 'herself',
    'it', 'its', 'itself',
    'they', 'them', 'themselves', 'their', 'theirs',
    'one', 'oneself',
    
    'this', 'that', 'these', 'those', 'such',
    'all', 'any', 'every', 'everyone', 'everybody', 'somebody', 'anybody', 'someone', 'anyone', 'everyone',
    'noone', 'nothing', 'none', 'nobody',
    'who', 'whom', 'what', 'where', 'when', 'why', 'how',
    'whoever', 'whomever', 'whatever', 'wherever', 'whenever', 'however'
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
    return 0 if pd.isna(subj) or not len(subj) else len(re.findall(r'[cfjqvxz]', subj))
    
def feature_fil_affix_sum(r):
    subj = r['word']
    return 0 if pd.isna(subj) or not len(subj) else afx.has_fil_affixing(subj)
    
def feature_ne_tag(r):
    subj = r['is_ne']
    return 0 if pd.isna(subj) or not len(subj) else 1

def is_fil_pronoun(word):
    return len(word) > 0 and word.lower() in fil_pronouns

def is_eng_pronoun(word):
    return len(word) > 0 and (word == 'I' or word.lower() in eng_pronouns)

def feature_pronoun_type(r):
    subj = r['word']
    if pd.isna(subj) or not len(subj):
        return 0
    if is_eng_pronoun(subj):
        return 2
    elif is_fil_pronoun(subj):
        return 1
    else:
        return 0

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
        return feature_pronoun_type
    elif n == 7:
        return feature_ne_tag
    else:
        raise Exception('feature out of bounds')

def main():
    # Interpret
    unprocessed_data = pd.read_csv('final_annotations.csv', dtype={'word': str}, keep_default_na=False, na_values=[], na_filter=False)
    ann1 = pd.read_csv('ann1.csv', dtype={'word': str, 'label': str, 'is_ne': str}, keep_default_na=False, na_values=[], na_filter=False)
    unprocessed_data = drop_irrelevant_columns(unprocessed_data)
    for ep in eng_pronouns:
        unprocessed_data.loc[len(unprocessed_data)] = [ep, 'ENG', '']
    for fp in fil_pronouns:
        unprocessed_data.loc[len(unprocessed_data)] = [fp, 'FIL', '']
    unprocessed_data = pd.concat([unprocessed_data, ann1], ignore_index=True, sort=False)
    unprocessed_data['previous_word'] = unprocessed_data['word'].shift(1, fill_value='')
    unprocessed_data['next_word'] = unprocessed_data['word'].shift(-1, fill_value='')
    for f in range(1, len(feature_columns)):
        unprocessed_data[feature_columns[f]] = unprocessed_data.apply(get_feature(f), axis=1)
    print(unprocessed_data[unprocessed_data['label'] == 'ENG'])
    
    # Test run
    #print(unprocessed_data[['word', 'isFirstLetterCapital', 'numVowels', 'wordLength', 'numNonPureAbakada', 'filAffixSum']].to_string())
    # To check na values (naalala ko na yung code)
    na_rows = unprocessed_data['word'].isna()
    #print(unprocessed_data[na_rows])
    #print("Rows with empty word: ", na_rows.sum())

    X = unprocessed_data[feature_columns + ['previous_word', 'next_word']]
    y = unprocessed_data['label']
    
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    #X_valtrain, X_valtest, y_valtrain, y_valtest = train_test_split(X_train, y_train, test_size=0.5, stratify=y_train)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
    X_valtrain, X_validation, y_valtrain, y_validation = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)
    
    # column transfomer to for the model to better understand the features and drop irrelevant columns
    
    # character N-grams for the word column (2-4)
    word_transformer = TfidfVectorizer(analyzer='char', ngram_range=(2,4))
    # numerical features scaling for good practice raw
    numerical_transformer = Pipeline(steps=[('scaler', QuantileTransformer())])
    preprocessor = ColumnTransformer(transformers=[('nGramText', word_transformer, 'word'), 
                                                    ('general_score', numerical_transformer, feature_columns[1:4] + feature_columns[6:]),
                                                    ('FIL_score', numerical_transformer, feature_columns[4:6])], remainder='drop')
    
    final_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', LogisticRegression(solver='saga', max_iter=2**15-1))]) # random_state=69

    # Training the model
    final_model.fit(X_validation, y_validation)

    # Evaluation of the model
    #print(X_valtest)
    y_predict = final_model.predict(X_validation)
    print(f"Prediction target test values: {y_predict}")
    print(f"Actual target test values:   {y_validation}")
    acc_score = accuracy_score(y_validation, y_predict)
    print(f"Model Accuracy on Test Set: {acc_score:.4f}")
    print(classification_report(y_validation, y_predict))
    #print(confusion_matrix(y_test, y_predict))
    
    # Custom prediction
    #prompt = ['marangya', 'ang', 'iyong', 'ugali', ',', 'kaya', '\'', 't', 'ikaw', 'ay', 'paparusahan', 'ng', 'Diyos', '-', 'Amang', 'makapangyarihan', '.']
    prompt = ['She\'s', 'not', 'that', 'good', 'at', 'all', ',', 'sa', 'totoo', 'lang', 'eh', '.']
    print(f"Prompt: {prompt}")
    print(f"Custom prediction results: {final_model.predict(wpar.pdfy(prompt))}")

    

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