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
import lightgbm as lgb

from enum import Enum
class ModelName(Enum):
    XGB = 1
    NBXGB = 2
    XGB_PERLABEL = 3
    NBXGB_PERLABEL = 4
    LGB = 5
    NBLGB = 6
    LGB_PERLABEL = 7
    NBLGB_PERLABEL = 8
    LOGREG = 9
    NBLOGREG = 10 # NBSVM
    LOGREG_PERLABEL = 11
    NBLOGREG_PERLABEL = 12
    LSVC = 13
    NBLSVC = 14
    LSVC_PERLABEL = 15
    NBLSVC_PERLABEL = 16
    RF = 17 # random forest
    ET = 18 # extra trees
    RNN = 19
    ONESVC = 20
    ONELOGREG = 21


from scipy.sparse import csr_matrix
class BaseLayerEstimator(ABC):
    
    @staticmethod
    def _calculate_nb(x, y):
        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)
        return csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
  
    @abstractmethod
    def train(self, x_train, y_train):
        """
        Params:
            x_train: np array
            y_train: pd series
        """
        pass
    
    @abstractmethod
    def predict(self, x_train):
        pass


from sklearn.calibration import CalibratedClassifierCV

class SklearnBLE(BaseLayerEstimator): 
    def __init__(self, clf, nb=False, seed=0, params={}, per_label_params={}, need_calibrated_classifier_cv=False):
        """
        Note: 
            1. If need to set params for different labels, let params={} when constructing
                so you can set seed, then use set_params() to set params per label
            2. For estimators like Linear SVC, CalibratedClassifierCV is needed
        """
        self.clf = clf
        self._nb = nb
        params['random_state'] = seed
        self.per_label_params = per_label_params
        self._seed = seed
        self.params = params
        self._need_calibrated_classifier_cv = need_calibrated_classifier_cv
            
    def set_params_for_label(self, label):
        """
        if need to set params for different labels, let params={} when constructing
        so you can set seed, and use this one to set params per label
        """
        self.params = self.per_label_params[label]
        self.params['random_state'] = self._seed

    def train(self, x, y):
        if self._nb:
            self._r = self._calculate_nb(x, y.values)
            x = x.multiply(self._r)
        if self._need_calibrated_classifier_cv:
            self.model = CalibratedClassifierCV(self.clf(**self.params))
        else:
            self.model = self.clf(**self.params)
        self.model.fit(x, y)

    def predict(self, x):
        if self._nb:
            x = x.multiply(self._r)
        return self.model.predict_proba(x)[:,1]
    
    def feature_importance(self):
        try:
            return self._clf.feature_importance
        except: #TODO: give a specific exception
            print('feature_importance not supported for this model')
            
            
    
    
class LightgbmBLE(BaseLayerEstimator):
    """
    You can instead use: SklearnBLE(LGBMClassifier)
    Unless yon need the early stopping function.
    """
    def __init__(self, params=None, nb=False, seed=0):
        self._nb = nb
        self._seed = seed
        self.set_params(params)
    
    def set_params(self, params):
        """
        if need to set params for different labels, let params={} when constructing
        so you can set seed, and use this one to set params per label
        """
        self.params = params
        self.params['data_random_seed'] = self._seed
    
    def train(self, x, y, valid_set_percent=0):
        """
        Params:
            x: np/scipy/ 2-d array or matrix
            y: pandas series
            valid_set_percent: (float, 0 to 1). 
                    0: no validation set. (imposible to use early stopping)
                    1: use training set as validation set (to check underfitting, and early stopping)
                    >0 and <1: use a portion of training set as validation set. (to check overfitting, and early stopping)
        
        """
        if self._nb:
            self._r = self._calculate_nb(x, y.values)
            x = x.multiply(self._r)
        
        if valid_set_percent != 0:
            if valid_set_percent > 1 or valid_set_percent < 0:
                raise ValueError('valid_set_percent must >= 0 and <= 1')
            if valid_set_percent != 1:
                x, x_val, y, y_val = train_test_split(x, y, test_size=valid_set_percent)


        lgb_train = lgb.Dataset(x, y)
        if valid_set_percent != 0:
            if valid_set_percent == 1:
                print('Evaluating using training set')
                self.model = lgb.train(self.params, lgb_train, valid_sets=lgb_train)
            else:
                lgb_val = lgb.Dataset(x_val, y_val)
                print('Evaluating using validation set ({}% of training set)'.format(valid_set_percent*100))
                self.model = lgb.train(self.params, lgb_train, valid_sets=lgb_val)
        else:
            print('No evaluation set, thus not possible to use early stopping. Please train with your best params.')
            self.model = lgb.train(self.params, lgb_train)
        
        
    def predict(self, x):
        if self._nb:
            x = x.multiply(self._r)
        if self.model.best_iteration > 0:
            print('best_iteration {} is chosen.'.format(best_iteration))
            result = self.model.predict(x, num_iteration=bst.best_iteration)
        else:
            result = self.model.predict(x)
        #print('predicting done')
        return result
            

from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

class RnnBLE(BaseLayerEstimator):
    """
    This class is still under development.
    """
    def __init__(self, window_length, n_features, label_cols, rnn_units=50, dense_units=50, dropout=0.1, 
                 mode='LSTM', bidirection=True):
        self._window_length = window_length
        self._n_features = n_features 
        self._label_cols = label_cols 
        self._rnn_units = rnn_units 
        self._dense_units = dense_units
        self._dropout = dropout
        self._mode = mode
        self._bidirection = bidirection
        self.init_model()
        
    def init_model(self, load_model=False, load_model_file=None):
        self._model = RnnBLE.get_lstm_model(self._window_length, self._n_features, self._label_cols, 
                                            self._rnn_units, self._dense_units, self._dropout, 
                                            self._mode, self._bidirection, load_model, load_model_file)
    
    @staticmethod
    def get_lstm_model(window_length, n_features, label_cols, rnn_units, dense_units, 
                       dropout, mode, bidirection, load_model, load_model_file):
        input = Input(shape=(window_length, n_features))
        rnn_layer = LSTM(rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
        if mode == 'GRU':
            rnn_layer = GRU(rnn_units, return_sequences=True, dropout=dropout, recurrent_dropout=dropout)
        if bidirection:
            x = Bidirectional(rnn_layer)(input)
        else:
            x = rnn_layer(input)
        x = GlobalMaxPool1D()(x)
        x = Dense(dense_units, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(len(label_cols), activation='sigmoid')(x)
        model = Model(inputs=input, outputs=x)
        
        if (load_model):
            print('load model: ' + str(load_model_file))
            model.load_weights(load_model_file)
        
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model 
    
    
    def train(self, epochs, x_train=None, y_train=None, batch_size=None, callbacks=None, 
              validation_split=0.0, validation_data=None, data_gen=None, 
              training_steps_per_epoch=None, load_model=False, load_model_file=None):
        if load_model:
            if load_model_file is None:
                raise ValueError('Since load model is True, please provide the load_model_file (path of the model)')
            else:
                self.init_model(load_model, load_model_file)
        if data_gen is None:
            if x_train is None or y_train is None or batch_size is None:
                raise ValueError('Since not training with data generator, please provide: x_train, y_train, batch_size')
            print('training without datagen')
            self._model.fit(x_train, y_train, batch_size=batch_size, validation_split=validation_split, epochs=epochs, callbacks=callbacks)
            return self._model # for chaining
        else:
            if training_steps_per_epoch is None:
                raise ValueError('training_steps_per_epoch can not be None when using data_gen')
            # with generator:
            print('training with datagen')
    
            self._model.fit_generator(
                generator=data_gen,
                steps_per_epoch=training_steps_per_epoch, 
                epochs=epochs, 
                validation_data=validation_data, # (x_val, y_val) 
                callbacks=callbacks
            )
            return self._model
            
    
    def predict(self, x, load_model_file=None):
        if load_model_file is not None:
            self._model.load_weights(load_model_file)
        return self._model.predict(x, verbose=1)#, batch_size=1024)
    
class OneVSOneRegBLE(BaseLayerEstimator):
    def __init__(self, x_train, y_train, model='logistic'):
        """
        x_train: sparse matrix, raw tfidf
        y_train: dataframe, with only label columns. should be 6 columns in total
        model: only support logistic or svc
        """
        self.r = {}
        self.setModelName(model)
        assert self.model_name in ['logistic', 'svc']
        self.param = {}
        self.param['logistic'] = {'identity_hate': 9.0,
                                     'insult': 1.5,
                                     'obscene': 1.0,
                                     'severe_toxic': 4.0,
                                     'threat': 9.0,
                                     'toxic': 2.7}
        self.param['svc'] = {'identity_hate': 0.9,
                             'insult': 0.15,
                             'obscene': 0.15,
                             'severe_toxic': 0.15,
                             'threat': 1.0,
                             'toxic': 0.29}
        
        
        
        for col in y_train.columns:
            #print('calculating naive bayes for {}'.format(col))
            self.r[col] = np.log(self.pr(1, y_train[col].values, x_train) / self.pr(0, y_train[col], x_train))
        #print('initializing done')
        #print('OneVsOne is using {} kernel'.format(self.model_name))
        
    def setModelName(self, name):
        self.model_name = name
        assert self.model_name in ['logistic', 'svc']
        #print('OneVsOne is using {} kernel'.format(self.model_name))
        
    def pr(self, y_i, y, train_features):
        p = train_features[np.array(y==y_i)].sum(0)
        return (p + 1) / (np.array(y == y_i).sum() + 1)
    
    def oneVsOneSplit(self, x_train, y_train, label):
        #print('Starting One vs One dataset splitting')
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        model_train = x_train[np.array(y_train == 1)]
        y_model_train = y_train[np.array(y_train == 1)]
        non_model_train = x_train[np.array(y_train == 0)]
        non_model_train = non_model_train[:model_train.shape[0]]
        y_non_model_train = y_train[np.array(y_train == 0)]
        y_non_model_train = y_non_model_train[:model_train.shape[0]]
        x_model_stack = vstack([model_train, non_model_train])
        y_model_stack = np.concatenate([y_model_train, y_non_model_train])
        x_nb = x_model_stack.multiply(self.r[label]).tocsr()
        y_nb = y_model_stack
        #print('splitting done!')
        return (x_nb, y_nb)
    
    def train(self, x_train, y_train, label):
        ### construct one vs one
        x_nb, y_nb = self.oneVsOneSplit(x_train, y_train, label)
        ### start training
        if self.model_name is 'logistic':
            #print('start training logistic regression')
            self.model = LogisticRegression(C=self.param['logistic'][label])
            self.model.fit(x_nb, y_nb)
            #print('training done')
            
        else:
            #print('start training linear svc regression')
            lsvc = LinearSVC(C=self.param['svc'][label])
            self.model = CalibratedClassifierCV(lsvc) 
            self.model.fit(x_nb, y_nb)
            #print('training done')
        

    
    def predict(self, x_test, label):
        #print('applying naive bayes to dataset')
        x_nb_test = x_test.multiply(self.r[label]).tocsr()
        #print('predicting')
        pred = self.model.predict_proba(x_nb_test)[:,1]
        #print('predicting done')
        return pred
    
##### example        
# aa = OneVSOneReg(train_tfidf, train[label_cols], model='logistic')
# aa.setModelName('svc')
# aa.train(train_tfidf,train['toxic'], 'toxic')
# aa.predict(test_tfidf, 'toxic')



    
class BaseLayerDataRepo():
    def __init__(self):
        self._data_repo = {}
        
    def add_tfidf_data(self, train_sentence, test_sentence, y_train, label_cols, compatible_models, 
                       word_ngram, word_max, word_min_df=1, word_max_df=1.0,
                       char_ngram=(0,0), char_max=100000, char_min_df=1, char_max_df=1.0):
        """
        Params:
            train_sentence: pd.Series. Usually the sentence column of a dataframe.
                e.g. train['comment_text]
            test_sentence: pd.Series. Usually the sentence column of a dataframe.
                e.g. test['comment_text]
            y_train: pd df, with columns names = label_cols
            label_cols: list of str. label column names
            compatible_models: list of ModelName. The intented models that will use this dataset.
                e.g. [ModelName.LGB, ModelName.LOGREG]
    
            word_ngram, word_max, word_min_df, word_max_df, char_ngram, char_max, char_min_df, char_max_df: tdidf params
            
        """
        # although label_cols can be extracted from y_train, including it in params can 
        # help make sure y_train is in right format. 
        assert len(list(y_train.columns)) == len(label_cols)
        assert set(list(y_train.columns)) - set(label_cols) == set()
        x_train, x_test, data_id = tfidf_data_process(train_sentence, test_sentence, 
                                                    word_ngram=(1,1), word_max=30000)
        self.add_data(data_id, x_train, x_test, y_train, label_cols, compatible_models)
        print('{} is added to the base layer data repo'.format(data_id))
    
    def add_data(self, data_id, x_train, x_test, y_train, label_cols, compatible_models, rnn_data=False):
        """
        x_train, x_test: nparray. use .values.reshape(-1,1) to convert pd.Series to nparray
        y_train: pd df, with columns names = label columns
        label_cols: list of str. label column names
        compatible_models: list of ModelName. The intented models that will use this dataset.
            e.g. [ModelName.LGB, ModelName.LOGREG]
        rnn_data: Boolean. Whether this data is for RNN
        """
        temp = {}
        
        temp['data_id'] = data_id
        temp['x_train'] = x_train
        temp['x_test'] = x_test
        temp['labes_cols'] = label_cols
        temp['compatible_models'] = set(compatible_models)
        
        if rnn_data: 
            temp['y_train'] = y_train # here y_train is a df
        else:
            label_dict = {}
            for col in label_cols:
                label_dict[col] = y_train[col]
            temp['y_train'] = label_dict # hence y_train is a dict with labels as keys
        
        self._data_repo[data_id] = temp
    
    def get_data(self, data_id):
        return self._data_repo[data_id]
    
    def remove_data(self, data_id):
        self._data_repo.pop(data_id, None)
        
    def get_compatible_models(self, data_id):
        return self._data_repo[data_id]['compatible_models']
    
    def remove_compatible_model(self, data_id, model_name):
        return self._data_repo[data_id]['compatible_models'].discard(model_name)
    
    def add_compatible_model(self, data_id, model_name):
        return self._data_repo[data_id]['compatible_models'].add(model_name)
                  
    def get_data_by_compatible_model(self, model_name):
        data_to_return = []
        for data_id in self._data_repo.keys():
            data = self._data_repo[data_id]
            if model_name in data['compatible_models']:
                data_to_return.append(data)
        return data_to_return
    
    def __len__(self):
        return len(self._data_repo)
    
    def __str__(self):
        output = ''
        for data_id in self._data_repo.keys():
            output+='data_id: {:20} \n\tx_train: {}\tx_test: {}\n\ty_train type: {}\n\tcompatible_models: {}\n '\
            .format(data_id, self._data_repo[data_id]['x_train'].shape, \
                    self._data_repo[data_id]['x_test'].shape, \
                    type(self._data_repo[data_id]['y_train']), \
                    self._data_repo[data_id]['compatible_models'])
        return output
    
    
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



def tfidf_data_process(train_sentence, test_sentence, word_ngram, word_max, word_min_df=1, word_max_df=1.0,
                       char_ngram=(0,0), char_max=100000, char_min_df=1, char_max_df=1.0):
    """
    Params:
        train_sentence: pd.Series. Usually the sentence column of a dataframe.
            e.g. train['comment_text]
        test_sentence: pd.Series. Usually the sentence column of a dataframe.
            e.g. test['comment_text]
        
        word_ngram, word_max, word_min_df, word_max_df, char_ngram, char_max, char_min_df, char_max_df: tdidf params

    return :x_train: sparse matrix
            y_train: DataFrame (containing all label columns)
            x_test: sparse matrix
            data_id: str, represents params
    """ 
    data_id = 'tfidf_word_{}_{}_{}_{}'.format(word_ngram, word_max, word_min_df, word_max_df)
    
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

    if char_ngram == (0,0):
        print('tfidf(word level) done')
        return (train_word, test_word, data_id)

    else:
        data_id = '{}_char_{}_{}_{}_{}'.format(data_id, char_ngram, char_max, char_min_df, char_max_df)

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
        train_char = char_vectorizer.transform(train_sentence.values)
        print('transforming test char')
        test_char = char_vectorizer.transform(test_sentence.values)

        x_train = hstack((train_char, train_word), format='csr')
        x_test = hstack((test_char, test_word), format='csr')

        print('tfidf(word & char level) done')
        return (x_train, x_test, data_id)
    
    
import pickle
def save_obj(obj, name, filepath):
    with open(filepath + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name, filepath):
    with open(filepath + name + '.pkl', 'rb') as f:
        return pickle.load(f)

import copy
class BaseLayerResultsRepo:
    def __init__(self, label_cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], 
                 load_from_file=True, filepath='obj/WithPreprocessedFile/'):
        """
        To start a new repo, set load_from_file to False, and give it a valid filepath so that files can be saved
        To load a save repo, set load_from_file to True, and give it the filepath
        """
        self._layer1_oof_train = {}
        self._layer1_oof_test = {}
        for label in label_cols:
            self._layer1_oof_train[label] = []
            self._layer1_oof_test[label] = []
        self._base_layer_est_preds = {}
        self._model_data_id_list = []
        self._base_layer_est_scores = {}
        self._label_cols = label_cols
        self.filepath = filepath
        self._save_lock = False # will be set to True if remove() is invoked successfully
        if load_from_file:
            print('load from file')
            self._layer1_oof_train = load_obj('models_layer1_oof_train', self.filepath)
            self._layer1_oof_test = load_obj('models_layer1_oof_test', self.filepath)
            self._base_layer_est_preds = load_obj('models_base_layer_est_preds', self.filepath)
            self._model_data_id_list = load_obj('models_model_data_id_list',self.filepath)
            self._base_layer_est_scores = load_obj('models_base_layer_est_scores',self.filepath)

    def get_model_data_id_list(self):
        return self._model_data_id_list
    
    def add(self, layer1_oof_train, layer1_oof_test, base_layer_est_preds, model_data_id_list):
        assert type(layer1_oof_train) == dict
        assert len(list(layer1_oof_train)) == len(self._label_cols)
        assert set(list(layer1_oof_train)) - set(self._label_cols) == set()
        assert type(layer1_oof_test) == dict
        assert len(list(layer1_oof_test)) == len(self._label_cols)
        assert set(list(layer1_oof_test)) - set(self._label_cols) == set()
        for label in self._label_cols:
            len(layer1_oof_train[label]) == len(layer1_oof_test[label]) == len(list(base_layer_est_preds))
        assert type(base_layer_est_preds) == dict
        assert type(model_data_id_list) == list
        assert set(list(base_layer_est_preds)) - set(model_data_id_list) == set()
        for model_data_id in model_data_id_list:
            if model_data_id in set(self._model_data_id_list):
                raise ValueError('{} is already in the repo'.format(model_data_id))
        for model_data_id in model_data_id_list:
            if model_data_id not in set(self._model_data_id_list):
                self._model_data_id_list.append(model_data_id)
                self._base_layer_est_scores[model_data_id] = 0
        for (key, values) in base_layer_est_preds.items():
            self._base_layer_est_preds[key] = values
        for label in self._label_cols:
            self._layer1_oof_train[label] += layer1_oof_train[label]
            self._layer1_oof_test[label] += layer1_oof_test[label]
    
    def add_score(self, model_data_id, score):
        assert score <= 1 and score >= 0
        if model_data_id not in set(self._model_data_id_list):
            raise ValueError('{} not in the repo. please add it first'.format(model_data_id))
        if model_data_id in set(self._model_data_id_list):
            print('{} already existed in the repo. score: {} update to {}'\
                  .format(model_data_id, self._base_layer_est_scores[model_data_id], score))
        self._base_layer_est_scores[model_data_id] = score
    
    def show_scores(self):
        """
        Returns:
            list of (name, score) tuple in sorted order by score
        """
        sorted_list_from_dict = sorted(self._base_layer_est_scores.items(), key=lambda x:x[1], reverse=True)
        for key, value in sorted_list_from_dict:
            print('{}\t{}'.format(value, key))
        return sorted_list_from_dict
    
    def get_results(self, threshold=None, chosen_ones=None):
        """
        Params:
            Note: threshold and chosen_ones can NOT both be not-None
            threshold: if not None, then return only ones that score >= threshold
            chosen_ones: list of model_data_id
        Returns: 
            layer1_oof_train, layer1_oof_test, base_layer_est_preds
        """
        if threshold != None and chosen_ones != None:
            raise ValueError('threshold and chosen_ones can NOT both be not-None')
        if threshold == None and chosen_ones == None:
            return self._layer1_oof_train, self._layer1_oof_test, self._base_layer_est_preds
        else:
            layer1_oof_train_temp = copy.deepcopy(self._layer1_oof_train) # copy only keep the keys, not the value reference
            layer1_oof_test_temp = copy.deepcopy(self._layer1_oof_test)   # deepcopy also keep the value reference
            base_layer_est_preds_temp = self._base_layer_est_preds.copy()
            base_layer_est_scores_temp = self._base_layer_est_scores.copy()
            model_data_id_list_temp = self._model_data_id_list.copy()
            if threshold != None:
                assert threshold <= 1 and threshold >= 0
                for (key, value) in base_layer_est_scores_temp.items():
                    if value < threshold:
                        self.remove(key)
            else: # chosen_ones != None
                assert type(chosen_ones) == list
                for model_data_id in model_data_id_list_temp:
                    if model_data_id not in chosen_ones:
                        self.remove(model_data_id)
                
            self._save_lock = False # not actually removed, so set it back to True

            r1, r2, r3 = self._layer1_oof_train, self._layer1_oof_test, self._base_layer_est_preds

            self._layer1_oof_train = layer1_oof_train_temp
            self._layer1_oof_test = layer1_oof_test_temp
            self._base_layer_est_preds = base_layer_est_preds_temp
            self._base_layer_est_scores = base_layer_est_scores_temp
            self._model_data_id_list = model_data_id_list_temp
            return r1, r2, r3
    
    def remove(self, model_data_id):
        #import pdb
        #pdb.set_trace()
        mdid_index = self._model_data_id_list.index(model_data_id)
        self._model_data_id_list.pop(mdid_index)
        self._base_layer_est_preds.pop(model_data_id)
        self._base_layer_est_scores.pop(model_data_id)
        for label in self._label_cols:
            self._layer1_oof_train[label].pop(mdid_index)
            self._layer1_oof_test[label].pop(mdid_index)
        self._save_lock = True
            
    def unlock_save(self):
        self._save_lock = False
            
    def save(self):
        if self._save_lock:
            print('save function is locked due to some results removed from the repo. \
            If you are sure about these changes, call unlock_save() to unlock the save function and save again.')
        else:
            save_obj(self._model_data_id_list, 'models_model_data_id_list', self.filepath)
            save_obj(self._layer1_oof_train, 'models_layer1_oof_train', self.filepath)
            save_obj(self._layer1_oof_test, 'models_layer1_oof_test', self.filepath)
            save_obj(self._base_layer_est_preds, 'models_base_layer_est_preds', self.filepath)
            save_obj(self._base_layer_est_scores, 'models_base_layer_est_scores', self.filepath)
            
            
from sklearn.model_selection import KFold, StratifiedKFold

def get_oof(clf, x_train, y_train, x_test, nfolds, stratified=False, shuffle=True, seed=1001):
    """
    Params:
        x_train, y_train, x_test
    """
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    oof_test_kf = np.empty((nfolds, ntest)) 
    if stratified:
        kf = StratifiedKFold(n_splits=nfolds, shuffle=shuffle, random_state=seed)
    else:
        kf = KFold(n_splits=nfolds, shuffle=shuffle, random_state=seed)

    for i, (tr_index, te_index) in enumerate(kf.split(x_train, y_train)):
        x_tr, x_te = x_train[tr_index], x_train[te_index]
        y_tr, y_te = y_train.iloc[tr_index], y_train.iloc[te_index]
        
        clf.train(x_tr, y_tr)
        oof_train[te_index] = clf.predict(x_te)
        oof_test_kf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_kf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)#, oof_test_kf.reshape(-1, nfolds)

def compute_layer1_oof(bldr, model_pool, label_cols, nfolds=5, seed=1001, sfm_threshold=None):
    """
    Params:
        bldr: an instance of BaseLayerDataRepo
        model_pool: dict. key: an option from ModelName. value: A model
        label_cols: list. names of labels
        nfolds: int 
        seed: int. for reproduce purpose
        sfm_threshold: str. e.g. 'median', '2*median'. if not None, then use SelectFromModel to select features
            that have importance > sfm_threshold
    Returns:
        layer1_est_preds: This the prediction of the layer 1 model_data, you can submit it to see the LB score
        layer1_oof_train: This will be used as training features in higher layers (one from each model_data)
        layer1_oof_mean_test: This will be used as testing features in higher layers (one from each model_data)
        model_data_id_list: This is the list of all layer 1 model_data
    """
    layer1_est_preds = {} # directly preditions from the base layer estimators # also layer1_oof_nofold_test

    layer1_oof_train = {}
    layer1_oof_mean_test = {}
    #layer1_oof_perfold_test = {}
    #layer1_oof_nofold_test = {}

    model_data_id_list = []

    for i, label in enumerate(label_cols):
    #     layer1_oof_train[label] = []
    #     layer1_oof_test[label] = []
        for model_name in model_pool.keys():
            for data in bldr.get_data_by_compatible_model(model_name):

                model_data_id = '{}_{}'.format(model_name, data['data_id'])
                current_run = 'label: {:12s} model_data_id: {}'.format(label, model_data_id)
                print('Computing... '+current_run)

                x_train = data['x_train']
                y_train = data['y_train'][label]
                x_test = data['x_test']

                NFOLDS = 4 # set folds for out-of-fold prediction
                #print(x_train.shape,y_train.shape,x_test.shape,label)

                if sfm_threshold is not None:
                    from sklearn.linear_model import LogisticRegression
                    from sklearn.feature_selection import SelectFromModel
                    model = LogisticRegression(solver='sag')
                    sfm = SelectFromModel(model, threshold=sfm_threshold)
                    print('dimension before selecting: train:{} test:{}'.format(x_train.shape, x_test.shape))
                    x_train = sfm.fit_transform(x_train, y_train)
                    x_test = sfm.transform(x_test)
                    print('dimension after selecting: train:{} test:{}'.format(x_train.shape, x_test.shape))

                model = model_pool[model_name]
                if 'PERLABEL' in str(model_name):
                    model.set_params_for_label(label)
    
                oof_train, oof_mean_test = get_oof(model, x_train, y_train, x_test, nfolds, seed)
                model.train(x_train, y_train)
                est_preds = model.predict(x_test)

                if label not in layer1_oof_train:
                    layer1_oof_train[label] = []
                    layer1_oof_mean_test[label] = []
                    #layer1_oof_perfold_test[label] = []
                    #layer1_oof_nofold_test[label] = []
                layer1_oof_train[label].append(oof_train)
                layer1_oof_mean_test[label].append(oof_mean_test)
                #layer1_oof_perfold_test[label].append(oof_perfold_test)
                #layer1_oof_nofold_test[label].append(est_preds.reshape(-1,1))

                if model_data_id not in layer1_est_preds:
                    layer1_est_preds[model_data_id] = np.empty((x_test.shape[0],len(label_cols)))
                    model_data_id_list.append(model_data_id)
                layer1_est_preds[model_data_id][:,i] = est_preds
    
    
    return layer1_est_preds, layer1_oof_train, layer1_oof_mean_test, model_data_id_list


def combine_layer_oof_per_label(layer1_oof_dict, label):
    """
    Util method for stacking
    """
    x = None
    data_list = layer1_oof_dict[label]
    for i in range(len(data_list)):
        if i == 0:
            x = data_list[0]
        else:
            x = np.concatenate((x, data_list[i]), axis=1)
    return x


def compute_layer2_oof(model_pool, layer2_inputs, train, label_cols, nfolds, seed):
    """
    Params:
        model_pool: dict. key: an option from ModelName. value: A model
        layer2_inputs: dict. key: an option from ModelName. value: chosen results from an instance of BaseLayerDataRepo
        train: pd. training data is required here to extract labels from it. For now, please make sure the train is not shuffled
            (TODO: THIS SHOULD BE INCLUDED IN layer2_inputs, which in terms should be included in BaseLayerDataRepo.)
        label_cols: list. names of labels
        nfolds: int
        seed: int. for reproduce purpose
    Returns:
        layer2_est_preds: This the prediction of the layer 1 model_data, you can submit it to see the LB score
        layer2_oof_train: This will be used as training features in higher layers (one from each model_data)
        layer2_oof_mean_test: This will be used as testing features in higher layers (one from each model_data)
        layer2_model_data_list: This is the list of all layer 1 model_data
    """
    layer2_est_preds = {} # directly preditions from the base layer estimators

    layer2_oof_train = {}
    layer2_oof_test = {}

    layer2_model_data_list = []

    for model_name in model_pool.keys():
        print('Generating Layer2 model {} OOF'.format(model_name))
        for i, label in enumerate(label_cols):
            #assert train.shape[0] == 159571

            model = model_pool[model_name]

            layer1_oof_train_loaded, layer1_oof_test_loaded, _ = layer2_inputs[model_name]

            x_train = combine_layer_oof_per_label(layer1_oof_train_loaded, label)
            x_test = combine_layer_oof_per_label(layer1_oof_test_loaded, label)

            oof_train, oof_test = get_oof(model,  x_train, train[label], x_test, nfolds, seed)

            if label not in layer2_oof_train:
                layer2_oof_train[label] = []
                layer2_oof_test[label] = []
            layer2_oof_train[label].append(oof_train)
            layer2_oof_test[label].append(oof_test)

            model_id = '{}_{}'.format(model_name, 'layer2')
            model.train(x_train, train[label])
            est_preds = model.predict(x_test)

            if model_id not in layer2_est_preds:
                layer2_est_preds[model_id] = np.empty((x_test.shape[0],len(label_cols)))
                layer2_model_data_list.append(model_id)
            layer2_est_preds[model_id][:,i] = est_preds
    
    return layer2_est_preds, layer2_oof_train, layer2_oof_test, layer2_model_data_list