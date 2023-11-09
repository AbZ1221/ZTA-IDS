import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
%matplotlib inline
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data.sampler import Sampler
from torch.utils.data import Dataset
import itertools
from collections import defaultdict
import preprocess

class balanced_sampler(Sampler):
    def __init__(self, data_source):
        self.label_dict         = data_source.label_dict

    def __iter__(self):
        dictkeys = list(self.label_dict.keys())
        dictkeys.sort()
        iter_list = []
        ## Data for each class
        key0 = dictkeys[0]
        data0    = self.label_dict[key0]
        pIndex0 = np.random.permutation(data0)
        key1 = dictkeys[1]
        data1    = self.label_dict[key1]
        pIndex1 = np.random.permutation(data1)
        key2 = dictkeys[2]
        data2    = self.label_dict[key2]
        pIndex2 = np.random.permutation(data2)
        key3 = dictkeys[3]
        data3    = self.label_dict[key3]
        pIndex3 = np.random.permutation(data3)
        key4 = dictkeys[4]
        data4    = self.label_dict[key4]
        pIndex4 = np.random.permutation(data4)
        key5 = dictkeys[5]
        data5    = self.label_dict[key5]
        pIndex5 = np.random.permutation(data5)
        # key6 = dictkeys[6]
        # data6    = self.label_dict[key6]
        # pIndex6 = np.random.permutation(data6)
        # key7 = dictkeys[7]
        # data7    = self.label_dict[key7]
        # pIndex7 = np.random.permutation(data7)
        # key8 = dictkeys[8]
        # data8    = self.label_dict[key8]
        # pIndex8 = np.random.permutation(data8)
        # key9 = dictkeys[9]
        # data9    = self.label_dict[key9]
        # pIndex9 = np.random.permutation(data9)
        # for i in range(max(len(pIndex0), len(pIndex1), len(pIndex2), len(pIndex3), len(pIndex4), len(pIndex5), len(pIndex6), len(pIndex7), len(pIndex8), len(pIndex9))):
        #     iter_list.append(np.random.permutation([pIndex0[i%len(pIndex0)], pIndex1[i%len(pIndex1)], pIndex2[i%len(pIndex2)], pIndex3[i%len(pIndex3)], pIndex4[i%len(pIndex4)], pIndex5[i%len(pIndex5)], pIndex6[i%len(pIndex6)], pIndex7[i%len(pIndex7)], pIndex8[i%len(pIndex8)], pIndex9[i%len(pIndex9)]]))
        for i in range(max(len(pIndex0), len(pIndex1), len(pIndex2), len(pIndex3), len(pIndex4), len(pIndex5))):
            iter_list.append(np.random.permutation([pIndex0[i%len(pIndex0)], pIndex1[i%len(pIndex1)], pIndex2[i%len(pIndex2)], pIndex3[i%len(pIndex3)], pIndex4[i%len(pIndex4)], pIndex5[i%len(pIndex5)]]))
        return iter(itertools.chain.from_iterable([iter for iter in iter_list]))

    def __len__(self):
        return len(self.data_source)
    
normalized_df, full_targ_data, test_cnt = preprocess.load_data()

class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        #Encoder
        self.encoder = nn.Sequential(
          nn.Linear(191, 128, bias=False),
          nn.Sigmoid(),
          nn.Linear(128, 64, bias=True)
        )

    def forward(self, x):
        x1 = self.encoder(x)
        return x1, None

model = autoencoder()
state_dict = torch.load('encoder_weights.pth', map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.cpu()


model.eval()

X_test = normalized_df.to_numpy(dtype='float')[test_cnt:,:]
X_train = normalized_df.to_numpy(dtype='float')[:test_cnt,:]
y_test = full_targ_data.values[len(test_cnt):]
y_train = full_targ_data.values[:len(test_cnt)]

label_list = [item for item in list(y_train.tolist())]
label_names = list(set(label_list))
label_names.sort()
label_maps = {name: i for i, name in enumerate(label_names)}

label_train = np.expand_dims(np.asarray([label_maps[item] for item in label_list]), 1)

label_list = [item for item in list(y_test.tolist())]
label_test = np.expand_dims(np.asarray([label_maps[item] for item in label_list]), 1)

Xy_train = np.concatenate((X_train, label_train), axis = 1)
Xy_test = np.concatenate((X_test, label_test), axis = 1)

batch_size = 32
train_loader = torch.utils.data.DataLoader(torch.from_numpy(Xy_train).float(), batch_size=batch_size, num_workers=2, shuffle=False)
test_loader = torch.utils.data.DataLoader(torch.from_numpy(Xy_test).float(), batch_size=batch_size, num_workers=2, shuffle=False)

mapped_feats = []
with torch.no_grad():
    for x in train_loader:
        x_ = x[:, :-1]
        y_ = x[:, -1]
        if torch.cuda.is_available():
            x_ = x_.cuda()

        # get output from the model, given the inputs
        feats, _ = model(x_)
        mapped_feats.append(torch.cat((feats.cpu(), y_.unsqueeze(1)), dim = 1))

training_feats_targ = torch.cat(mapped_feats, dim = 0)

mepped_feats = []
with torch.no_grad():
    for x in test_loader:
        x_ = x[:, :-1]
        y_ = x[:, -1]
        if torch.cuda.is_available():
            x_ = x_.cuda()

        # get output from the model, given the inputs
        feats, _ = model(x_)
        mepped_feats.append(torch.cat((feats.cpu(), y_.unsqueeze(1)), dim = 1))

testing_feats_targ = torch.cat(mepped_feats, dim = 0)

from typing import Optional

class MultiheadAttention(nn.Module):
    """Multihead Self Attention module
    Args:
        embed_dim (int): Total dimension of the model.
        num_heads (int): The number of heads.
        dropout (float, optional):
            Dropout probabiliry on attn_output_weights. Default: ``0.0``
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        out_dim: int = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        head_dim = embed_dim // num_heads
        if head_dim * num_heads != embed_dim:
            raise ValueError(f"`embed_dim ({embed_dim})` is not divisible by `num_heads ({num_heads})`")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = torch.nn.Dropout(dropout)
        self.head_dim = head_dim

        self.scaling = self.head_dim ** -0.5

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        if out_dim is not None:
            self.out_proj = nn.Linear(embed_dim, out_dim, bias=True)
        else:
            self.out_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (Tensor): shape: ``[batch_size, sequence_length, embed_dim]``.
            attention_mask (Tensor or None, optional):
                shape: ``[batch_size, 1, sequence_length, sequence_length]``
        Returns:
            Tensor: The resulting tensor. shape: ``[batch, sequence_length, embed_dim]``
        """
        if x.ndim != 3 or x.shape[2] != self.embed_dim:
            raise ValueError(
                f"The expected input shape is (batch, sequence, embed_dim=={self.embed_dim}). " f"Found {x.shape}."
            )
        batch_size, length, embed_dim = x.size()
        if attention_mask is not None:
            shape_ = (batch_size, 1, length, length)
            if attention_mask.size() != shape_:
                raise ValueError(f"The expected attention mask shape is {shape_}. " f"Found {attention_mask.size()}.")

        shape = (batch_size, length, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd
        k = self.k_proj(x).view(*shape).permute(0, 2, 3, 1)  # B, nH, Hd, L
        v = self.v_proj(x).view(*shape).transpose(2, 1)  # B, nH, L, Hd

        weights = self.scaling * (q @ k)  # B, nH, L, L
        if attention_mask is not None:
            weights += attention_mask

        weights = torch.nn.functional.softmax(weights, dim=-1)
        weights = self.dropout(weights)

        output = weights @ v  # B, nH, L, Hd
        output = output.transpose(2, 1).reshape(batch_size, length, embed_dim)

        output = self.out_proj(output)
        return output
    

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 128, 11)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(128, 128, 9)
        self.conv3 = nn.Conv1d(128, 256, 7)
        self.conv4 = nn.Conv1d(256, 256, 5)
        #self.attn = nn.Conv1d(64, 9, 9)
        self.mhatt = MultiheadAttention(embed_dim = 256, num_heads = 8, out_dim = 64)
        self.fc1 = nn.Linear(64, 10)
        #self.fc4 = nn.Linear(32, 10)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        x = torch.sigmoid(x)
        x = self.pool1(self.lrelu(self.conv1(x)))
        x = self.lrelu(self.conv2(x))
        x = self.lrelu(self.conv3(x))
        x = self.lrelu(self.conv4(x))

        x = self.mhatt(x.transpose(2,1)).transpose(2,1)

        aligned = x.mean(-1)

        x = self.fc1(aligned)
        return aligned, x


net = Net()
if torch.cuda.is_available():
    net.cuda()

learningRate = 1e-3
criterion = torch.nn.CrossEntropyLoss() # Train classifier
#optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=5e-3, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=learningRate)

class data_emo(Dataset):
    def __init__(self, data) -> None:
        super().__init__()
        label_dict = defaultdict(list)
        self.data = data
        for i, item in enumerate(data):
            item = int(item[-1])
            label_dict[item].append(i)
        self.label_dict = label_dict

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        return self.data[index]
    
train_dataset = data_emo(training_feats_targ)
emo_sampler = balanced_sampler(train_dataset)

batch_size = 32
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=emo_sampler, num_workers=2, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_feats_targ, batch_size=128, num_workers=2, shuffle=False)

maxcor = 0
epochs = 300

##### For GPU #######
if torch.cuda.is_available():
    net.cuda()

net.train()
for epoch in range(epochs):
    total_loss = []
    net.train()
    for x in tqdm(train_loader):
        x_ = x[:, :-1]
        y_ = x[:, -1].long()
        if torch.cuda.is_available():
            x_ = x_.cuda()
            y_ = y_.cuda()

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        _, outputs = net(x_.unsqueeze(1))

        # get loss for the predicted output
        loss = criterion(outputs, y_)
        # get gradients w.r.t to parameters
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        # update parameters
        optimizer.step()
    net.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for x in test_loader:
            x_ = x[:, :-1]
            y_ = x[:, -1].long()
            if torch.cuda.is_available():
                x_ = x_.cuda()
                y_ = y_.cuda()

            # get output from the model, given the inputs
            _, outputs = net(x_.unsqueeze(1))

            # get loss for the predicted output
            loss = criterion(outputs, y_)
            total_loss += [loss.item()]
            _, predicted = torch.max(outputs.data, 1) # arg_max(output)
            total += y_.size(0)
            correct += (predicted == y_).sum().item()
    if correct > maxcor:
        maxcor = correct
        torch.save(net.state_dict(), 'cnnatt_weights.pth')

    print('Accuracy of the network on test traces: %2.2f %%' % (
        100 * correct / total))
    print('epoch {}, loss {}'.format(epoch, np.average(total_loss)))
