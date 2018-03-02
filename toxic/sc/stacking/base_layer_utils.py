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
    LGB = 3
    NBLGB = 4
    LOGREG = 5
    NBSVM = 6
    NBLSVC = 7
    RF = 8 # random forest
    RNN = 9
    ONESVC = 10
    ONELOGREG = 11


class BaseLayerEstimator(ABC):
    
    def _pr(self, y_i, y, train_features):
        p = train_features[np.array(y==y_i)].sum(0)
        return (p + 1) / (np.array(y == y_i).sum() + 1)
    
    def _nb(self, x_train, y_train):
        assert isinstance(y_train, pd.DataFrame)
        r = {}
        for col in y_train.columns:
            print('calculating naive bayes for {}'.format(col))
            r[col] = np.log(self._pr(1, y_train[col].values, x_train) / self._pr(0, y_train[col], x_train))
        return r
    
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
            print('calculating naive bayes for {}'.format(col))
            self.r[col] = np.log(self.pr(1, y_train[col].values, x_train) / self.pr(0, y_train[col], x_train))
        print('initializing done')
        print('OneVsOne is using {} kernel'.format(self.model_name))
        
    def setModelName(self, name):
        self.model_name = name
        assert self.model_name in ['logistic', 'svc']
        print('OneVsOne is using {} kernel'.format(self.model_name))
        
    def pr(self, y_i, y, train_features):
        p = train_features[np.array(y==y_i)].sum(0)
        return (p + 1) / (np.array(y == y_i).sum() + 1)
    
    def oneVsOneSplit(self, x_train, y_train, label):
        print('Starting One vs One dataset splitting')
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
        print('splitting done!')
        return (x_nb, y_nb)
    
    def train(self, x_train, y_train, label):
        ### construct one vs one
        x_nb, y_nb = self.oneVsOneSplit(x_train, y_train, label)
        ### start training
        if self.model_name is 'logistic':
            print('start training logistic regression')
            self.model = LogisticRegression(C=self.param['logistic'][label])
            self.model.fit(x_nb, y_nb)
            print('training done')
            
        else:
            print('start training linear svc regression')
            lsvc = LinearSVC(C=self.param['svc'][label])
            self.model = CalibratedClassifierCV(lsvc) 
            self.model.fit(x_nb, y_nb)
            print('training done')
        

    
    def predict(self, x_test, label):
        print('applying naive bayes to dataset')
        x_nb_test = x_test.multiply(self.r[label]).tocsr()
        print('predicting')
        pred = self.model.predict_proba(x_nb_test)[:,1]
        print('predicting done')
        return pred
    
##### example        
# aa = OneVSOneReg(train_tfidf, train[label_cols], model='logistic')
# aa.setModelName('svc')
# aa.train(train_tfidf,train['toxic'], 'toxic')
# aa.predict(test_tfidf, 'toxic')



class SklearnBLE(BaseLayerEstimator):
    def __init__(self, clf, seed=0, params=None):
        params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict_proba(x)[:,1]

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from scipy import sparse
from sklearn.calibration import CalibratedClassifierCV
class NbSvmBLE(BaseLayerEstimator, BaseEstimator, ClassifierMixin):
    def __init__(self, mode=ModelName.NBSVM, seed=0, params=None):
        self._mode = mode
        params['random_state'] = seed
        self._params = params


    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        #return self._clf.predict(x.multiply(self._r))
        return self._clf.predict_proba(x.multiply(self._r))[:,1] # chance of being 1 ([:,0] chance of being 0)

    def fit(self, x, y):
        # Check that X and y have correct shape
        y = y.values
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y==y_i].sum(0)
            return (p+1) / ((y==y_i).sum()+1)

        self._r = sparse.csr_matrix(np.log(pr(x,1,y) / pr(x,0,y)))
        x_nb = x.multiply(self._r)
        #self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=self.n_jobs).fit(x_nb, y)
        self._clf = LogisticRegression(**self._params).fit(x_nb, y)
        if self._mode == ModelName.NBLSVC:
            self._clf = CalibratedClassifierCV(LinearSVC(**self._params)).fit(x_nb, y)

        return self
    
    def train(self, x_train, y_train):
        self.fit(x_train, y_train)
    
    def feature_importance(self):
        return self._clf.feature_importance

    
class XgbBLE(BaseLayerEstimator):
    def __init__(self, seed=0, params=None):
        self.param = params
        self.param['seed'] = seed
        self.nrounds = params.pop('nrounds', 250)

    def train(self, x_train, y_train):
        dtrain = xgb.DMatrix(x_train, label=y_train)
        self.gbdt = xgb.train(self.param, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict_proba(xgb.DMatrix(x))[:,1]

from sklearn.feature_selection import SelectFromModel

class LightgbmBLE(BaseLayerEstimator):
    def __init__(self, x_train, y_train, params=None, nb=True, seed=0):
        """
        constructor:

            x_train: should be a np/scipy/ 2-d array or matrix. only be used when nb is true
            y_train: should be a dataframe
            
        Example:
            ll = LightgbmBLE(train_tfidf, train[label_cols], params=params, nb=True)
            result = pd.DataFrame()
            for col in label_cols:
                    print(col)
                    ll.train(train_tfidf, train[col], col)
                    result[col] = ll.predict(test_tfidf, col)
        """
        #### check naive bayes
        if nb:
            print('Naive Bayes is enabled')
            self.r = self._nb(x_train, y_train)
        else:
            print('Naive Bayes is disabled')
            self.r = None
        ##### set values    
        self.nb = nb
        self.set_params(params)
        print('LightgbmBLE is initialized')
    
    
    def set_params(self, params):
        self.params = params
    
    
    
    def _pre_process(self, x_train, y_train, label=None):
        if self.nb:
            if label is None:
                raise ValueError('Naive Bayes is enabled. label cannot be None.')
            print('apply naive bayes to feature set')
            x = x_train.multiply(self.r[label])
            if isinstance(x_train, csr_matrix):
                x = x.tocsr()
        else:
            x = x_train
        if isinstance(y_train, pd.Series):
            y = y_train.values
        else:
            y = y_train
        return (x, y)
    
    
    def train(self, x_train, y_train, label=None):
        x, y = self._pre_process(x_train, y_train, label)
        lgb_train = lgb.Dataset(x, y)
        lgb_eval = lgb.Dataset(x, y, reference=lgb_train)
        self.model = lgb.train(self.params, lgb_train, valid_sets=lgb_eval, verbose_eval=20)
        
        
    def predict(self, x_train, label=None):
        x, _ = self._pre_process(x_train, y_train=None, label=label)
        print('starting predicting')
        result = self.model.predict(x)
        print('predicting done')
        return result
        
            

from keras.layers import Dense, Embedding, Input, LSTM, Bidirectional, GlobalMaxPool1D, Dropout, BatchNormalization
from keras.models import Model

class RnnBLE(BaseLayerEstimator):
    def __init__(self, window_length, n_features, label_cols, rnn_units=50, dense_units=50, dropout=0.1, mode='LSTM', bidirection=True, batch_size=32, epochs=2):
        self._model = RnnBLE.get_lstm_model(window_length, n_features, label_cols, rnn_units, dense_units, dropout, mode, bidirection)
        self._batch_size = batch_size
        self._epochs = epochs
        
    @staticmethod
    def get_lstm_model(window_length, n_features, label_cols, rnn_units, dense_units, dropout, mode, bidirection):
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
        model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        return model 
    
    
    def train(self, x_train, y_train):
        self._model.fit(x_train, y_train, batch_size=self._batch_size, epochs=self._epochs)
        
    
    def predict(self, x):
        return self._model.predict(x)#, batch_size=1024)
    
    
    
class BaseLayerDataRepo():
    def __init__(self):
        self._data_repo = {}
    
    def add_data(self, data_id, x_train, x_test, y_train, label_cols, compatible_model=[ModelName.LOGREG], rnn_data=False):
        """
        x_train, x_test: ndarray
        y_train: pd df
        """
        temp = {}
        
        temp['data_id'] = data_id
        temp['x_train'] = x_train
        temp['x_test'] = x_test
        temp['labes_cols'] = label_cols
        temp['compatible_model'] = set(compatible_model)
        
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
        
    def get_compatible_model(self, data_id):
        return self._data_repo[data_id]['compatible_model']
    
    def remove_compatible_model(self, data_id, model_name):
        return self._data_repo[data_id]['compatible_model'].discard(model_name)
    
    def add_compatible_model(self, data_id, model_name):
        return self._data_repo[data_id]['compatible_model'].add(model_name)
                  
    def get_data_by_compatible_model(self, model_name):
        data_to_return = []
        for data_id in self._data_repo.keys():
            data = self._data_repo[data_id]
            if model_name in data['compatible_model']:
                data_to_return.append(data)
        return data_to_return
    
    def __len__(self):
        return len(self._data_repo)
    
    def __str__(self):
        output = ''
        for data_id in self._data_repo.keys():
            output+='data_id: {:20} \n\tx_train: {}\tx_test: {}\n\ty_train type: {}\n\tcompatible_model: {}\n '\
            .format(data_id, self._data_repo[data_id]['x_train'].shape, \
                    self._data_repo[data_id]['x_test'].shape, \
                    type(self._data_repo[data_id]['y_train']), \
                    self._data_repo[data_id]['compatible_model'])
        return output