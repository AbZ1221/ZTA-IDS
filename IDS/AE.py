import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import preprocess

normalized_df = preprocess.load_data()
X_train, X_test = train_test_split(normalized_df, test_size=0.1)

#Prepare data loaders
batch_size = 32
train_loader = torch.utils.data.DataLoader(torch.from_numpy(X_train.values).float(), batch_size=batch_size, num_workers=2)
test_loader = torch.utils.data.DataLoader(torch.from_numpy(X_test.values).float(), batch_size=batch_size, num_workers=2)

#Define the Autoencoder
class autoencoder(nn.Module):
    def __init__(self):
        super(autoencoder, self).__init__()
        #Encoder
        self.encoder = nn.Sequential(
          nn.Linear(191, 128, bias=False),
          nn.Sigmoid(),
          nn.Linear(128, 64, bias=True),
          nn.Sigmoid()
        )
        #Decoder
        self.decoder = nn.Sequential(
          nn.Linear(64, 128, bias=True),
          nn.Sigmoid(),
          nn.Linear(128, 191, bias=True),
          nn.Sigmoid()
        )
    def forward(self, x):
        x1 = self.encoder(x)
        # import ipdb; ipdb.set_trace()
        x2 = self.decoder(x1)
        return x1, x2

model = autoencoder()

criterion = torch.nn.MSELoss() # mean squared error (Linear Regression)
#optimizer = torch.optim.SGD(model.parameters(), lr=learningRate, weight_decay=5e-3, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)


epochs = 1000

##### For GPU #######
if torch.cuda.is_available():
    model.cuda()

# Iterate over the epochs
for epoch in range(epochs):
    total_loss = []
    model.train()
    for x in tqdm(train_loader):
        # Converting inputs and labels to Variable
        if torch.cuda.is_available():
            x = x.cuda()

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        _, outputs = model(x)

        # get loss for the predicted output
        loss = criterion(outputs, x)
        # get gradients w.r.t to parameters
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # update parameters
        optimizer.step()
    model.eval()
    with torch.no_grad():
        for x in test_loader:
            if torch.cuda.is_available():
                x = x.cuda()

      # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
            optimizer.zero_grad()

      # get output from the model, given the inputs
            _, outputs = model(x)

      # get loss for the predicted output
            loss = criterion(outputs, x)
            total_loss += [loss.item()]
    print('epoch {}, loss {}'.format(epoch, np.average(total_loss)))


total_loss = []
model.cuda()
model.eval()
with torch.no_grad():
    for x in test_loader:
        if torch.cuda.is_available():
            x = x.cuda()

        # Clear gradient buffers because we don't want any gradient from previous epoch to carry forward, dont want to cummulate gradients
        optimizer.zero_grad()

        # get output from the model, given the inputs
        _, outputs = model(x)

        # get loss for the predicted output
        loss = criterion(outputs, x)
        total_loss += [loss.item()]
print('loss {}'.format(np.average(total_loss)))

torch.save(model.state_dict(), 'encoder_weights.pth')

