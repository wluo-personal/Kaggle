import pickle
def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

import copy
class BaseLayerResultsRepo:
    def __init__(self, label_cols=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'], load_from_file=True):
        self._layer1_oof_train = {}
        self._layer1_oof_test = {}
        for label in label_cols:
            self._layer1_oof_train[label] = []
            self._layer1_oof_test[label] = []
        self._base_layer_est_preds = {}
        self._model_data_id_list = []
        self._base_layer_est_scores = {}
        self._label_cols = label_cols
        self._save_lock = False # will be set to True if remove() is invoked successfully
        if load_from_file:
            print('load from file')
            self._layer1_oof_train = load_obj('13models_layer1_oof_train')
            self._layer1_oof_test = load_obj('13models_layer1_oof_test')
            self._base_layer_est_preds = load_obj('13models_base_layer_est_preds')
            self._model_data_id_list = load_obj('13models_model_data_id_list')
            self._base_layer_est_scores = load_obj('13models_base_layer_est_scores')

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
                for model_data_id in chosen_ones:
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
            Call unlock_save() to unlock the save function and save again.')
        else:
            save_obj(self._model_data_id_list, '13models_model_data_id_list')
            save_obj(self._layer1_oof_train, '13models_layer1_oof_train')
            save_obj(self._layer1_oof_test, '13models_layer1_oof_test')
            save_obj(self._base_layer_est_preds, '13models_base_layer_est_preds')
            save_obj(self._base_layer_est_scores, '13models_base_layer_est_scores')