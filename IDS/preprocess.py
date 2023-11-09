import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline

# Data Processing Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Pipeline Testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

from sklearn.metrics import recall_score as recall
from sklearn.metrics import precision_score as precision
from sklearn.metrics import f1_score as fscore

from pickle import dump, load


#for example 135 columns from proto to 5
def reduce_column(s, to_keep):
    '''
    Reduces the string values found in a column
    to the values provided in the list 'to_keep'.
    ---
    Input:
        s: string
        to_keep: list of strings
    Returns:
        string, s if s should be kept, else 'other'
    '''
    s = s.lower().strip()
    if s not in to_keep:
        return 'other'
    else:
        return s
    

def load_data():
    dfs = []
    path = 'UNSW_NB15_testing-set.csv'
    testing_csv = pd.read_csv(path)
    dfs.append(testing_csv)

    path = 'UNSW_NB15_training-set.csv'
    training_csv = pd.read_csv(path)
    dfs.append(training_csv)

    all_data = pd.concat(dfs).reset_index(drop=True)


    # The columns below, besides 'attack_cat' which acts as an optional label for multiclass, are
    # not needed for further modeling after some research online as to what they mean.
    binary_cols = ['trans_depth', 'ct_srv_src',	'ct_state_ttl',	'ct_dst_ltm',	'ct_src_dport_ltm',	'ct_dst_sport_ltm',	'ct_dst_src_ltm',	'ct_src_ltm', 'ct_srv_dst']
    droppable_cols = ['id','attack_cat','label', 'response_body_len', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'is_sm_ips_ports']

    # Define the individual steps
    ohe_step = ('ohe', OneHotEncoder(sparse=False))
    ssc_step = ('std_sclr', StandardScaler())
    log_step = ('log10', FunctionTransformer(np.log10))

    # Make the step part of a pipeline
    ohe_pipe = Pipeline([ohe_step])
    ssc_pipe = Pipeline([ssc_step])
    log_ssc_pipe = Pipeline([log_step, ssc_step])

    # Columns to transform: categorical columns for encoding, numeric feature columns for standardizing
    ohe_cols = ['proto', 'state', 'service']
    non_ssc_cols = ohe_cols+droppable_cols+binary_cols
    #ssc_cols = all_data.drop(columns = ohe_cols+droppable_cols+['label']).columns
    ssc_cols = [col for col in all_data.columns if col not in non_ssc_cols]

    # Transformer input: tuple w/ contents ('name', SomeTransformer(Parameters), columns)
    transformer = [
        ('one_hot_encoding', ohe_pipe, ohe_cols),
        ('standard_scaling', ssc_pipe, ssc_cols)
    ]

    ct = ColumnTransformer(transformers=transformer, remainder='passthrough')


    # Recreating column labels for one_hot_encoded data
    cat_cols = np.concatenate((np.sort(all_data.proto.unique()),
                            np.sort(all_data.state.unique()),
                            np.sort(all_data.service.unique())))

    # Combining transformed column labels with non-transformed column labels.
    # Order matters here: transformed columns in order, non-transformed features, target column
    new_cols =  np.concatenate((cat_cols, ssc_cols, binary_cols))

    all_data.drop(all_data[all_data['attack_cat'] == 'Analysis'].index, inplace = True)
    all_data.drop(all_data[all_data['attack_cat'] == 'Backdoor'].index, inplace = True)
    all_data.drop(all_data[all_data['attack_cat'] == 'Shellcode'].index, inplace = True)
    all_data.drop(all_data[all_data['attack_cat'] == 'Worms'].index, inplace = True)

    full_targ_data = all_data.attack_cat.astype('category')

    full_input_data = pd.DataFrame(ct.fit_transform(all_data.drop(columns=droppable_cols)))
    full_input_data.columns = new_cols

    normalized_df=(full_input_data-full_input_data.min())/(full_input_data.max()-full_input_data.min())
    return normalized_df, full_targ_data, testing_csv, 
