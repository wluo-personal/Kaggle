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




def tfidf_data_process(word_ngram, word_max, word_min_df=1, word_max_df=1.0,
                       char_ngram=(0,0), char_max=100000, char_min_df=1, char_max_df=1.0):
    """
    Params:
        char_ngram: (0,0) means not processing at char level

    return :x_train: sparse matrix
            y_train: DataFrame
            x_test: sparse matrix
    """
    data_id = 'wordtfidf_word_{}_{}_{}_{}_char_{}_{}_{}_{}'.format(word_ngram, word_max, word_min_df, word_max_df,
                                           char_ngram, char_max, char_min_df, char_max_df)

    
    train = pd.read_csv('/home/kai/data/wei/Toxic/models/data/cleaned_train.csv')
    test = pd.read_csv('/home/kai/data/wei/Toxic/models/data/cleaned_test.csv')
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

    train_sentence = train['comment_text_cleaned_polarity']
    test_sentence = test['comment_text_cleaned_polarity']

    train_sentence_retain_punctuation = train['comment_text_cleaned_retain_punctuation']
    test_sentence_retain_punctuation = test['comment_text_cleaned_retain_punctuation']
    print('loading data done!')
    #########################################
    word_vectorizer = TfidfVectorizer(ngram_range=word_ngram, #1,3
                                        strip_accents='unicode',
                                        max_features=word_max,
                                        min_df = word_min_df,
                                        max_df = word_max_df,
                                        analyzer='word',
                                        stop_words='english',
                                        sublinear_tf=True,
                                        token_pattern=r'\w{1,}')
    print('fitting word')
    word_vectorizer.fit(train_sentence.values)
    print('transforming train word')
    train_word = word_vectorizer.transform(train_sentence.values)
    print('transforming test word')
    test_word = word_vectorizer.transform(test_sentence.values)

    y_train = train[label_cols]

    if char_ngram == (0,0):
        return (train_word, y_train, test_word, data_id)

    else:
        char_vectorizer = TfidfVectorizer(ngram_range=char_ngram,  #2,5
                                          strip_accents='unicode',
                                          max_features=char_max, #200000
                                          min_df = char_min_df,
                                          max_df = char_max_df,
                                          analyzer='char',
                                          sublinear_tf=True)

        print('fitting char')
        char_vectorizer.fit(train_sentence_retain_punctuation.values)
        print('transforming train char')
        train_char = char_vectorizer.transform(train_sentence_retain_punctuation.values)
        print('transforming test char')
        test_char = char_vectorizer.transform(test_sentence_retain_punctuation.values)

        x_train = hstack((train_char, train_word), format='csr')
        x_test = hstack((test_char, test_word), format='csr')

        return (x_train, y_train, x_test, data_id)
