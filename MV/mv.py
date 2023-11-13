from CryptoPAn_fun import CryptoPAn
import ipaddress
import numpy as np
c = CryptoPAn(('\x00\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\x0c\r\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1e').encode())
import pickle
import pandas as pd # import pandas for loading, processing and working with csv files simpler
from tqdm import tqdm # tqdm is for showing and logging

dfs1 = pd.read_csv('./UNSW-NB15_1.csv', header = None) # Read the data, It does not have any header
dfs2 = pd.read_csv('./UNSW-NB15_2.csv', header = None) # Read the data, It does not have any header
dfs3 = pd.read_csv('./UNSW-NB15_3.csv', header = None) # Read the data, It does not have any header
dfs4 = pd.read_csv('./UNSW-NB15_4.csv', header = None) # Read the data, It does not have any header
df_col = pd.read_csv('./NUSW-NB15_features.csv', encoding='ISO-8859-1') # Read the features, its encoding type is ISO

dfs = pd.concat([dfs1, dfs2, dfs3, dfs4])

df_col['Name'] = df_col['Name'].apply(lambda x: x.strip().replace(' ', '').lower()) # make all names lowercase for simplicity and ignoring mistakes
dfs.columns = df_col['Name'] # add colomn names of the dfs file to its header
dfs.head() # write down 5 top records to check if it's OK

ips = dfs.get('srcip').values # get the values of the secip and write it to ips variable

dfs.sport = dfs.sport.fillna(value=0).replace('-', 0)

ano_ips = [] # empty list to fill anonymized ips in
for ip in tqdm(ips):
  ano_ips.append(c.anonymize(ip))

# usage is by integer indexes for example ano_ips[0] will returns the first element of anonymized ips

dfs['srcipan'] = ano_ips

cp = CryptoPAn("".join([chr(x) for x in range(0,32)]).encode())
ano_ports = []
sport_int = []
for pid in tqdm(range(len(dfs.sport))):
  port = dfs.sport.iloc[pid]
  try:
    port = str(int(port))
  except:
    port = '0'
  sport_int.append(int(port))
  ano_ports.append(cp.anonymize(port))

dfs.sport = sport_int
dfs['sportan'] = ano_ports



distinct_ips = list(set(ano_ips)) # Make ips unique
distinct_ips.sort() # Sorting the ips is ascending

# First Partitioning method, based on ip decimal values
def partitioning(n):
  decimal_ips = [int(ipaddress.ip_address(ip)) for ip in distinct_ips] # convert ip to decimal
  ranges = [int(i*100/n) for i in range(1,n)]
  thresh = np.percentile(decimal_ips, ranges)
  parts = {i:[] for i in range(n)} # create a dictionary with partition numbers as keys and empty values

  thresh = np.concatenate([[0], thresh, [max(decimal_ips)+1]])
  
  for i in range(n):
    for dec in decimal_ips:
      if thresh[i] <= dec and dec < thresh[i+1]: # check if the ip value is between start and end point add it to the partion i
        parts[i] += [ipaddress.ip_address(dec).__str__()]

  return parts


n = 10 # Set the number of partitions for method 1
partitionedips_1 = partitioning(n) # for example call partitioning first method with n=3

# partition based on ports
distinct_ports = list(set(ano_ports)) # Make ips unique
distinct_ports.sort() # Sorting the ips is ascending

def partitioning_2(n):
  decimal_ports = [int(prt) for prt in distinct_ports] # convert ip to decimal
  ranges = [int(i*100/n) for i in range(1,n)]
  thresh = np.percentile(decimal_ports, ranges)
  parts = {i:[] for i in range(n)} # create a dictionary with partition numbers as keys and empty values
  
  thresh = np.concatenate([[0], thresh, [max(decimal_ports)+1]])
  
  for i in range(n):
    for dec in decimal_ports:
      if thresh[i] <= dec and dec <= thresh[i+1]: # check if the ip value is between start and end point add it to the partion i
        parts[i] += [str(dec)]

  return parts

partitionedips_1_2 = partitioning_2(n)

data_partitioned_1 = [] # define empty list for data partioned using the first method
for p in list(partitionedips_1.keys()): # iterating over the partition ids
  d1 = [] # create a temp list to fill it with the values of ip addresses with the ip value of zeros position
  for ip in partitionedips_1[p]: # iterate over the ips of p-th partition
    d2 = dfs.loc[dfs['srcipan']==ip].copy() #find the next ip values
    d1.append(d2) # append new ip values to d1
  data_partitioned_1.append(d1) # add temp list to the list of all data

counter = 0
for item1 in data_partitioned_1:
  for item2 in item1:
    counter += item2.shape[0]

data_partitioned_2 = [] # define empty list for data partioned using the first method
for p in list(partitionedips_1_2.keys()): # iterating over the partition ids
  d1 = [] # create a temp list to fill it with the values of ip addresses with the ip value of zeros position
  for ip in partitionedips_1_2[p]: # iterate over the ips of p-th partition
    d2 = dfs.loc[dfs['sportan']==ip].copy() #find the next ip values
    d1.append(d2) # append new ip values to d1
  data_partitioned_2.append(d1) # add temp list to the list of all data

counter = 0
for item1 in data_partitioned_2:
  for item2 in item1:
    counter += item2.shape[0]

data_partitioned_1_2 = {} # define empty list for data partioned using the first method
for k1 in tqdm(partitionedips_1.keys()):
  for k2 in tqdm(partitionedips_1_2.keys()): # iterating over the partition ids
    sqet1 = dfs.loc[(dfs['srcipan'].isin(partitionedips_1[k1]) & dfs['sportan'].isin(partitionedips_1_2[k2]))].copy()
    #data_partitioned_1_2.append(d1) # add temp list to the list of all data
    data_partitioned_1_2[k1,k2] = sqet1

count_p12 = 0
for _, item in data_partitioned_1_2.items():
  count_p12 += item.shape[0]
count_p12


file1 = open("partitioned1_twice.pkl", "wb")
pickle.dump(data_partitioned_1_2, file1)
file1.close()

data_partitioned_2 = [] # define empty list for data partioned using the second method
for p in distinct_ips: # iterating over the partition ids / distinct ips
  data_partitioned_2.append(dfs.loc[dfs['srcip']==p]) # find and add the data records with the same srcip / srcip equals to p

V = np.random.randint(1,5,n) # Create random generator
V_ = np.random.randint(1,5,n) # Create random generator

r = 2 # for example

## TODO: Is 50 constant? What about r?
V0 = [50]*n - r*V # calculate V0
V0_ = [50]*n - r*V_ # calculate V0

print(V0)

test = data_partitioned_1_2.items()

data_partitioned_1_2 = [(p1, part1) for p1, part1 in data_partitioned_1_2.items()]
data_partitioned_1_2 = data_partitioned_1_2[6:]
record_counter = 0
for key in data_partitioned_1_2:
  record_counter += len(key)
  print(record_counter)



# !cp '/gdrive/My Drive/data_part_1_trace1.pkl' ./
# file1 = open("data_part_1_trace1.pkl", "rb")
# data_part_1_trace1, p_ = pickle.load(file1)
# file1.close()
# print(p_)
# data_partitioned_1_2 = [(p1, part1) for p1, part1 in data_partitioned_1_2.items()]
# data_partitioned_1_2 = data_partitioned_1_2[6:]
# import ipdb;ipdb.set_trace()
#print(p)
data_part_1_trace1 = {}
for p, part in data_partitioned_1_2: #data_partitioned_1_2.items(): # Iterating over the partitions
  print(f'generating partition {p}')
  ip_tmp = [] # make it free for each partition
  port_tmp = []
  for pid in tqdm(range(len(part['sportan']))): # Iterating over the records of the selected partition
    port = dfs.sportan.iloc[pid]
    for j in range(V0_[p[1]]): # run anonymization V0 of partition number p times over the selected port
      port = c.anonymize(port)
    port_tmp.append(port)

    ip = dfs.srcipan.iloc[pid]
    for j in range(V0[p[0]]): # run anonymization V0 of partition number p times over the selected ip
      ip = c.anonymize(ip)
    ip_tmp.append(ip)
  print(len(part))
  print(len(ip_tmp))
  print(len(port_tmp))
  data_part_1_trace1[p] = part # add each partition to data_part_1_trace1 variable
  data_part_1_trace1[p]['sportant1'] = port_tmp
  data_part_1_trace1[p]['srcipant1'] = ip_tmp
  
  file1 = open("data_part_1_trace1.pkl", "wb")
  pickle.dump((data_part_1_trace1, p),file1)
  file1.close()

file1 = open("data_part_1_trace2.pkl", "rb")
data_partitioned_1_2 = pickle.load(file1)
file1.close()

si = {}
sp = {}

keys = list(data_partitioned_1_2.keys())
for i in range(len(keys)):
    si[0] = data_partitioned_1_2[keys[i]]['srcipan']
    sp[0] = data_partitioned_1_2[keys[i]]['sportan']
    si[1] = data_partitioned_1_2[keys[i]]['srcipant1']
    sp[1] = data_partitioned_1_2[keys[i]]['sportant1']
    si[2] = data_partitioned_1_2[keys[i]]['srcipant2']
    sp[2] = data_partitioned_1_2[keys[i]]['sportant2']
    data_partitioned_1_2[keys[i]].drop(columns=['srcipan','srcipant1','srcipant2','sportan','sportant1','sportant2'], inplace=True)
    data_partitioned_1_2[keys[i]]['srcipan1'] = si[i%3]
    data_partitioned_1_2[keys[i]]['srcipan2'] = si[(i+1)%3]
    data_partitioned_1_2[keys[i]]['srcipan3'] = si[(i+2)%3]
    data_partitioned_1_2[keys[i]]['sportan1'] = sp[i%3]
    data_partitioned_1_2[keys[i]]['sportan2'] = sp[(i+1)%3]
    data_partitioned_1_2[keys[i]]['sportan3'] = sp[(i+2)%3]
    data_partitioned_1_2[keys[i]]['real'] = i%3


import pandas as pd
data12 = pd.concat(data_partitioned_1_2.values())

data12.drop(columns = ['srcip','sport'], inplace=True)

data12.to_csv('20percent_mv.csv', columns=data12.columns)

file1 = open("partitioned1_twice.pkl", "rb")
data_partitioned_1_2 = pickle.load(file1)
file1.close()

file1 = open("VS.pkl", "rb")
V, V0, V_, V0_ = pickle.load(file1)
file1.close()

ano1 = []
for part, datapart in enumerate(data_part_1_trace1):
  for data in datapart:
    ano1.append(data)

compare_ips = pd.DataFrame(columns = ['partition', 'srcip', 'trcip']) # Create an empty dataframe with the required columns for src and trc ips

for key in data_part_1_trace1.keys(): # iterating over source ip partitions, and also over seed traced ips
  partSrc, partTrc = data_partitioned_1_2[key], data_part_1_trace1[key]
  #srclist = partSrc.values.tolist() # convert the partition from pandas.dataframe to list for simplicity
  for src, trc in zip(partSrc, partTrc): # iterating over ips of source and trace
    compare_ips = compare_ips.append({'partition': p, 'srcip': src[0], 'trcip': trc[0]}, ignore_index=True) #append new row to the compare_ips dataframe

compare_ips = pd.DataFrame(columns = ['partition', 'srcip', 'trcip']) # Create an empty dataframe with the required columns for src and trc ips

for key in data_part_1_trace1.keys(): # iterating over source ip partitions, and also over seed traced ips
  partSrc, partTrc = data_partitioned_1_2[key], data_part_1_trace1[key]
  #srclist = partSrc.values.tolist() # convert the partition from pandas.dataframe to list for simplicity
  for src, trc in zip(partSrc, partTrc): # iterating over ips of source and trace
    compare_ips = compare_ips.append({'partition': p, 'srcip': src[0], 'trcip': trc[0]}, ignore_index=True) #append new row to the compare_ips dataframe

compare_ips.to_csv('part1.csv', columns=['partition', 'srcip', 'trcip'])

data_part_1_trace2 = {} # a free list to fill with the new seed trace values of anonymizied ips V times

for p in data_part_1_trace1: # Iterating over the partitions of the first traced
  print(f'generating partition {p}')
  port_tmp = []
  ip_tmp = []
  part = data_part_1_trace1[p]
  for pid in tqdm(range(len(part['sportant1']))): # Iterating over the records of the selected partition
    port = part.sportant1.iloc[pid]
    for j in range(V_[p[1]]): # run anonymization V0 of partition number p times over the selected ip
      port = c.anonymize(port)
    port_tmp.append(port)
    ip = part.srcipant1.iloc[pid] # get the ip from the selected record
    for j in range(V[p[0]]): # run anonymization V0 of partition number p times over the selected ip
      ip = c.anonymize(ip)
    ip_tmp.append(ip)

  data_part_1_trace2[p] = part # add each partition to data_part_1_trace1 variable
  data_part_1_trace2[p]['sportant2'] = port_tmp
  data_part_1_trace2[p]['srcipant2'] = ip_tmp

file1 = open("data_part_1_trace2.pkl", "wb")
pickle.dump(data_part_1_trace2, file1)
file1.close()

record_counter = 0
for key in data_part_1_trace2.keys():
  record_counter += len(data_part_1_trace2[key])
  print(record_counter)


record_counter = 0
for key in data_part_1_trace1.keys():
  record_counter += len(data_part_1_trace1[key])
  print(record_counter)


n = len(data_partitioned_2) # Get the number of partitions for method 2

V = np.random.randint(1,5,n) # Create random generator

print(V)

r = 2 # for example

V0 = [50]*n - r*V # calculate V0

print(V0)

data_part_2_trace1 = [] # a free list to fill with the seed trace from output of anonymized ips V0 times

for p, part in enumerate(data_partitioned_2): # Iterating over the partitions
  print(f'generating view for partition {p}')
  part_tmp = [] # make it free for each partition
  for rec in tqdm(list(part.to_records(index=False))): # Iterating over the records of the selected partition
    rec = rec.tolist() # convert record type to list in order to work with simpler
    ip = rec[0] # get tge ip from the selected record
    for j in range(V0[p]): # run anonymization V0 times over the selected ip
      ip = c.anonymize(ip)
    part_tmp.append([ip] + list(rec[1:])) # Gather the anonymized ip along with the other columns for each partion
  data_part_2_trace1.append(part_tmp) # add each partition to data_part_2_trace1 variable


text = ''
for p2 in data_part_2_trace1[0]: #iterating over records of 0th partition change 0 with the other values until n
  for v in p2:
    text += str(v) + ' , '
  text += '\n'
print(text)

data_partitioned_2_ = data_partitioned_2[:2]
data_part_2_trace1 = data_part_2_trace1[:2]

compare_ips = pd.DataFrame(columns = ['partition', 'srcip', 'trcip']) # Create an empty dataframe with the required columns for src and trc ips

for p, (partSrc, partTrc) in enumerate(zip(data_partitioned_2, data_part_2_trace1)): # iterating over source ip partitions, and also over seed traced ips
  srclist = partSrc.values.tolist() # convert the partition from pandas.dataframe to list for simplicity
  for src, trc in zip(srclist, partTrc): # iterating over ips of source and trace
    compare_ips = compare_ips.append({'partition': p, 'srcip': src[0], 'trcip': trc[0]}, ignore_index=True) #append new row to the compare_ips dataframe


data_part_2_trace2 = [] # a free list to fill with the new seed trace values of anonymizied ips V times

for p, part in enumerate(data_part_2_trace1): # Iterating over the partitions
  print(f'generating partition {p}')
  part_tmp = [] # make it free for each partition
  for rec in tqdm(part): # Iterating over the records of the selected partition
    ip = rec[0] # get the ip from the selected record
    for j in range(V[p]): # run anonymization V of partition number p times over the selected ip
      ip = c.anonymize(ip)
    part_tmp.append([ip] + list(rec[1:])) # Gather the anonymized ip along with the other columns for each partion
  data_part_2_trace2.append(part_tmp) # add each partition to data_part_2_trace variable

# fill the column names for the final table into variable column_names
column_names = ['view 1', 'view 2', 'view 3'] # define 3 new columns names
for name in data_partitioned_2[0].columns[1:]: # add the colomn names from the original table but the first column (srcip -> view 2)
  column_names.append(name)

multiview = pd.DataFrame(columns = column_names) # define an empty table with colomn names defined above

for srcsc, trac1, trac2 in zip(data_partitioned_2, data_part_2_trace1, data_part_2_trace2): #iterate over the partitions of original table, trace1 table, and trace2 table at the same time
  src = srcsc.values.tolist() # convert the original table into a simpler structure
  for sr,t1,t2 in zip(src, trac1, trac2): # iterate over the records (rows) of the tables selected for this partition
    view = [t1[0]] + [c.anonymize(sr[0])] + t2 # append the columns i.e. trace1 srcip trace2 etc...
    dic_tmp = {}
    for n, v in zip(column_names, view): # add view into dictionary structure to be added to pandas dataframe
      dic_tmp[n] = v
    multiview = multiview.append(dic_tmp, ignore_index=True) # append generated view to multiview

headers = ["view 1", "view 2", "view 3", "attack_cat", "label"]

multiview.to_csv("views.csv", columns=headers)

multiview.to_csv("views_others.csv")

