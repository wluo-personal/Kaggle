import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

def scatter_per_label(preds_repo, draw_label, first_n=-1):
    """
    Params:
        Note: to see the plot, add this line: %matplotlib inline
        preds_repo: (dict)  key: prediction name (usually sth like 'xxxxx.csv') 
                            value: prediction df (usually read from saved csv)
        draw_label: (str)   the lable to draw scatter plot
        fisrt_n:    (int)   only select the first n entries
    """
    pred1 = list(preds_repo.values())[0]
    label_cols = list(pred1.columns)
    label_cols.pop(0) # remove 'id'
    if draw_label not in label_cols:
        raise ValueError('available labels: {}'.format(label_cols))

    preds_mats = []
    names = []
    for label in label_cols:
        preds_list = []
        print(label)
        for filename, value in preds_repo.items():
            if filename not in names:
                names.append(filename)
            preds_list.append(value[label].values.reshape(-1,1))
        preds_mats.append(np.hstack(preds_list))

    print(names)
    assert len(preds_mats) == len(label_cols)
    assert preds_mats[0].shape[1] == len(preds_repo)
    
    if first_n != -1:
        number_of_entries = preds_mats[0].shape[0] # the shape is commonly: (153164, 2)
        if first_n > number_of_entries:
            raise ValueError('the max value of first_n is: ' + str(number_of_entries))
        temp = pd.DataFrame(preds_mats[label_cols.index(draw_label)][:first_n], columns=names)
    else:
        temp = pd.DataFrame(preds_mats[label_cols.index(draw_label)], columns=names)
    scatter_matrix(temp, figsize=(14,14))