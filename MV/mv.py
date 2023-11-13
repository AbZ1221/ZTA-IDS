import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
# Data Processing Imports
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

# Pipeline Testing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score

import pickle



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
    

file1 = open("data_part_1_trace2.pkl", "rb")
data_part_1_trace2 = pickle.load(file1)
file1.close()

dfs = []
for key in data_part_1_trace2.keys():
  dfs.append(data_part_1_trace2[key])
all_data = pd.concat(dfs).reset_index(drop=True)


## Column cleaning steps: some of the CSV's leave the point blank for zero values.
## This results in Pandas loading in NaN values in columns where it otherwise expects numeric values. 
# Fill all NaN attack categories w/ value: 'normal'
all_data['attack_cat'] = all_data.attack_cat.fillna(value='normal').apply(lambda x: x.strip().lower())
# Replace blank spaces with zero
all_data['ct_ftp_cmd'] = all_data.ct_ftp_cmd.replace(to_replace=' ', value=0).astype(int)
# Replace NaN with zero
all_data['ct_flw_http_mthd'] = all_data.ct_flw_http_mthd.fillna(value=0)
# Replace NaN with zero and all values > 0 with 1
all_data['is_ftp_login'] = (all_data.is_ftp_login.fillna(value=0) >0).astype(int)

## Reduce categorical features into smaller sets:
## Ex: 135 unique values in `proto` become "tcp", "udp", "arp", "unas", and "other"
transformations = {
    'proto':['tcp', 'udp', 'arp', 'unas'],
    'state':['fin', 'con', 'int'],
    'service':['-', 'dns', 'http', 'ftp-data', 'smtp', 'ftp', 'ssh']
}
for col, keepers in transformations.items():
    all_data[col] = all_data[col].apply(reduce_column, args=(keepers,))


def ipsplitor(s):
  return s.strip().split('.')

ipcolumns = ['srcipan', 'srcipant1', 'srcipant2', 'dstip']

for column in ipcolumns:
  splited = all_data[column].apply(ipsplitor)
  part1 = []
  part2 = []
  part3 = []
  part4 = []
  for item in splited:
    part1.append(item[0])
    part2.append(item[1])
    part3.append(item[2])
    part4.append(item[3])
  all_data[column+'1'] = part1
  all_data[column+'2'] = part2
  all_data[column+'3'] = part3
  all_data[column+'4'] = part4


keep_cols = [
       'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes',
       'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload',
       'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz',
       'dmeansz', 'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime',
       'ltime', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
       'is_sm_ips_ports', 'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login',
       'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm',
       'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm', 'attack_cat', 'label',
       'sportan', 'sportant1', 'sportant2', 'srcipan1', 'srcipan2', 'srcipan3', 'srcipan4', 'srcipant11', 'srcipant12', 'srcipant13', 'srcipant14', 'srcipant21', 'srcipant22', 'srcipant23', 'srcipant24'
]

droppable_cols = ['srcipan', 'srcipant1', 'srcipant2', 'dstip', 'srcip', 'sport']

all_data = all_data.drop(columns = droppable_cols)

# The columns below, besides 'attack_cat' which acts as an optional label for multiclass, are
# not needed for further modeling after some research online as to what they mean.
binary_cols = ['is_sm_ips_ports', 'is_ftp_login']
#droppable_cols.append('attack_cat') # Ignore another target column in this processing
target_cols = ['label', 'attack_cat']

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
non_ssc_cols = ohe_cols+droppable_cols+binary_cols+target_cols
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
new_cols =  np.concatenate((cat_cols, ssc_cols, binary_cols, target_cols))

for col in ssc_cols[1:]:
  print(col)
  pd.to_numeric(all_data[col])

for col in ['dsport']:
  all_data[col] = pd.to_numeric(all_data[col], errors='coerce').fillna(0)

#pd.DataFrame(ct.fit_transform(all_data.drop(columns=['attack_cat']))).head()
new_data = pd.DataFrame(ct.fit_transform(all_data))
new_data.columns = new_cols

