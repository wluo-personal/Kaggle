import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score
from scipy.sparse import csr_matrix, hstack, vstack




def tfidf_data_process(word_ngram,char_ngram,char_max_df=1,skipgram=False):
    """
    wei/Toxic/models/data/cleaned_train.csv
    wei/Toxic/models/data/cleaned_test.csv

    return :x_train: sparse matrix
            y_train: DataFrame
            x_test: sparse matrix
    """
    train = pd.read_csv('/home/kai/data/wei/Toxic/models/data/cleaned_train.csv')
    test = pd.read_csv('/home/kai/data/wei/Toxic/models/data/cleaned_test.csv')
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    train_sentence = train['comment_text_cleaned_polarity']
    test_sentence = test['comment_text_cleaned_polarity']

    train_sentence_retain_punctuation = train['comment_text_cleaned_retain_punctuation']
    test_sentence_retain_punctuation = test['comment_text_cleaned_retain_punctuation']
    print('loading data done!')
    #########################################
    if stop_words:
        phrase_vectorizer = TfidfVectorizer(ngram_range=word_ngram, ###1,3
                                    strip_accents='unicode', 
                                    max_features=100000, 
                                    analyzer='word',
                                    sublinear_tf=True,
                                    stop_words='english',
                                    token_pattern=r'\w{1,}')
        char_vectorizer = TfidfVectorizer(ngram_range=(2,5),  ###2,5
                                      strip_accents='unicode', 
                                      max_features=200000, 
                                      analyzer='char', 
                                      stop_words='english',
                                      sublinear_tf=True)
    else:
        phrase_vectorizer = TfidfVectorizer(ngram_range=(1,3), ###1,3
                                    strip_accents='unicode', 
                                    max_features=100000, 
                                    analyzer='word',
                                    sublinear_tf=True,
                                    token_pattern=r'\w{1,}')
        char_vectorizer = TfidfVectorizer(ngram_range=(2,5),  ###2,5
                                      strip_accents='unicode', 
                                      max_features=200000, 
                                      analyzer='char', 
                                      sublinear_tf=True)
    

    print('fitting char')
    char_vectorizer.fit(train_sentence_retain_punctuation.values)
    print('fitting phrase')
    phrase_vectorizer.fit(train_sentence.values)


    print('transforming train char')
    train_char = char_vectorizer.transform(train_sentence_retain_punctuation.values)
    print('transforming train phrase')
    train_phrase = phrase_vectorizer.transform(train_sentence.values)


    print('transforming test char')
    test_char = char_vectorizer.transform(test_sentence_retain_punctuation.values)
    print('transforming test phrase')
    test_phrase = phrase_vectorizer.transform(test_sentence.values)


    x_train = hstack((train_char, train_phrase), format='csr')
    x_test = hstack((test_char, test_phrase), format='csr')
    y_train = train[label_cols]
    idd = 'wordtfidf_ng13_mf10w_chartfidf_ng25_mf20w_real'
    
    return (x_train, y_train, x_test, idd)


