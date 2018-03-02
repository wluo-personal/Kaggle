from fastText import load_model
import re, os
import numpy as np
import pandas as pd
import gc

def normalize(s):
    """
    Given a text, cleans and normalizes it. Feel free to add your own stuff.
    """
    #s = s.lower()
    # Replace ips
    #s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    #s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    #s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    return s

def text_to_vector(text, window_length, n_features, ft_model):
    """
    Given a string, normalizes it, then splits it into words and finally converts
    it to a sequence of word vectors.
    """
    text = normalize(text)
    words = text.split()
    window = words[-window_length:]
    
    x = np.zeros((window_length, n_features))

    for i, word in enumerate(window):
        x[i, :] = ft_model.get_word_vector(word).astype('float32')

    return x

def df_to_data(df, window_length, n_features, ft_model):
    """
    Convert a given dataframe to a dataset of inputs for the NN.
    """
    x = np.zeros((len(df), window_length, n_features), dtype='float32')

    for i, comment in enumerate(df['comment_text_cleaned'].values):
        x[i, :] = text_to_vector(comment, window_length, n_features, ft_model)

    return x

def fasttext_data_process(window_length=200, shuffle_train=False, val_ratio=0, first_n_entries=-1):
    """
    Params:
        window_length: (int, 200 to 450) the max number of words you choose in a text
        shuffe_train: (boolean) whether you want to shuffle training data
        val_raio: (float, 0 to 1.0) the validation set portion sepereated from training set
        first_n_entries: (int -1 to max length of training data) only choose the first n entries
                        of the training and testing data, mainly for test purpose
                                
    Returns:
        x_train, x_test, x_val: numpy array
        y_train, y_val: pd dataframe        
    """
    print('\nLoading data')
    train = pd.read_csv('/home/kai/data/haoyan/ToxicClassificationCopy3/data/cleaned_train.csv')
    test = pd.read_csv('/home/kai/data/haoyan/ToxicClassificationCopy3/data/cleaned_test.csv')
    if first_n_entries != -1:
        max_value = min(len(train), len(test))
        if first_n_entries < max_value:
            train = train[:first_n_entries]
            test = test[:first_n_entries]
        else:
            raise ValueError('max value of first_n_entries is ' + str(max_value))
    
    label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    print('train shape: {}. test shape: {}'.format(train.shape, test.shape))
    
    print('\nLoading FT model')
    ft_model = load_model('/home/kai/data/resources/FastText/wiki.en.bin')
    n_features = ft_model.get_dimension()

    print(n_features)
    #window_length = 200 # The amount of words we look at per example. Experiment with this.
    print('window: {}. dimension(n_features): {}'.format(window_length, n_features))
    
    split_index = len(train)
    if val_ratio > 0:
        split_index = round(split_index * (1-val_ratio)) 
    if shuffle_train:
        train = train.sample(frac=1)
    
    df_train = train.iloc[:split_index]
    x_train = df_to_data(df_train, window_length, n_features, ft_model)
    y_train = df_train[label_cols]
    
    df_val = train.iloc[split_index:]
    # Convert validation set to fixed array
    x_val = df_to_data(df_val, window_length, n_features, ft_model)
    y_val = df_val[label_cols]
    # Get test data ready
    x_test = df_to_data(test, window_length, n_features, ft_model)
    
    del ft_model, train, test
    
    return x_train, y_train, x_test, x_val, y_val