from google.colab import drive # layout for google drive

drive.mount('/content/drive') # mounting point for google drive

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
%matplotlib inline

# Use GPU
torch.cuda.is_available()
device = torch.device("cuda:0")
print(device)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("Running on GPU")
else:
    device = torch.device("cpu")
    print("Running on CPU")

data = np.load("/content/drive/My Drive/Molecular docking/training_data.npy", allow_pickle=True)
# data = np.load("/content/drive/My Drive/Molecular docking/training_data_1000.npy", allow_pickle=True)
"""
data[i][0] - receptor
data[i][1] - ligand
data[i][2] - label
 """

# HYPERPARAMETERS
RECEPTOR_DIM = 256
LIGAND_DIM = 30
LEARNING_RATE = 0.025 # Check optimizer first
MOMENTUM = 0.70 # For SGD
BATCH_SIZE = 128
EPOCHS = 500 # 215

X_receptor = torch.Tensor([i[0] for i in data]).view(-1, 1, RECEPTOR_DIM, RECEPTOR_DIM)
X_ligand = torch.Tensor([i[1] for i in data]).view(-1, LIGAND_DIM * LIGAND_DIM)
y = torch.Tensor([i[2] for i in data])

print("Receptors: ", len(X_receptor), "shape: ", X_receptor.shape)
print("Ligands: ", len(X_ligand), "shape: ", X_ligand.shape)
print("Labels: ", len(y))


# HYBRID NET, FC
class TwoInputsNet(nn.Module):
    def __init__(self):
        global RECEPTOR_DIM, LIGAND_DIM
        super(TwoInputsNet, self).__init__()
        # Set convolution layers for receptors
        self.conv1 = nn.Conv2d(1, 32, 10)
        self.conv2 = nn.Conv2d(32, 64, 10)
        self.conv3 = nn.Conv2d(64, 128, 10)
        self.pool1 = nn.MaxPool2d((10, 10))
        
        #Set fully connected layers for receptors
        self.fc1 = nn.Linear(4608, 1024) # Calculate input value first
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 64)
        # self.fc4 = nn.Linear(256, 128)
        # self.fc5 = nn.Linear(128, 128)
        # self.fc6 = nn.Linear(128, 64)
        
        # Set fully connected layers for ligands
        self.fc10 = nn.Linear(LIGAND_DIM * LIGAND_DIM, 512) # Calculate input value first
        self.fc11 = nn.Linear(512, 128)
        self.fc12 = nn.Linear(128, 64)
        self.fc13 = nn.Linear(64, 64)
        
        # After concatenation
        self.fc20 = nn.Linear(128, 64)
        self.fc21 = nn.Linear(64, 64)
        self.fc22 = nn.Linear(64, 2)
        #self.fc23 = nn.Linear(2, 2)

    def forward(self, input1, input2): # (receptor - r, ligand - l)
        # Receptors
        r = F.relu(self.conv1(input1))
        r = self.pool1(r)
        r = F.relu(self.conv2(r))
        r = F.relu(self.conv3(r))
        # r = F.relu(self.conv4(r))
        # r = F.relu(self.conv5(r))
        # r = F.relu(self.conv6(r))
                
        r = r.view(r.size(0), -1)
        r = F.relu(self.fc1(r))
        r = F.relu(self.fc2(r))
        r = F.relu(self.fc3(r))
        # r = F.relu(self.fc4(r))
        # r = F.relu(self.fc5(r))
        # r = F.relu(self.fc6(r))
        
        # Ligands        
        l = F.relu(self.fc10(input2))
        l = F.relu(self.fc11(l))
        l = F.relu(self.fc12(l))
        l = F.relu(self.fc13(l))
        
        # now we can reshape `c` and `f` to 2D and concatenate them
        combined = torch.cat((r, l), dim=1)
        combined = torch.cat((r.view(r.size(0), -1), l.view(l.size(0), -1)), dim=1)
        out = F.relu(self.fc20(combined))
        out = F.relu(self.fc21(out))
        out = self.fc22(out)
        
        return F.softmax(out, dim=1)
    
net = TwoInputsNet().to(device)
print(net)

# Hybrid train
# Define optimizer
# optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
# optimizer = optim.SGD(net.parameters(), momentum=MOMENTUM, lr=LEARNING_RATE)
optimizer = optim.Adadelta(net.parameters(), lr=0.5, rho=0.9, eps=1e-06, weight_decay=0)
loss_function = nn.MSELoss()

VAL_PCT = 0.1
val_size = int(len(X_receptor) * VAL_PCT)

print("Validation size: ", val_size)

# Define number of training and test samples
train_X_receptor = X_receptor[:-val_size]
train_X_ligand = X_ligand[:-val_size]
train_y = y[:-val_size]

test_X_receptor = X_receptor[-val_size:]
test_X_ligand = X_ligand[-val_size:]
test_y = y[-val_size:]

print("Number of training receptors: ", len(train_X_receptor))
print("Number of training ligands: ", len(train_X_ligand))
print("Number of testing receptors: ", len(test_X_receptor))
print("Number of testing ligands: ", len(test_X_ligand))


def train(net):
    global RECEPTOR_DIM, LIGAND_DIM, EPOCHS, BATCH_SIZE
    print("begin the training")
    losses = []
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X_receptor), BATCH_SIZE)):
            batch_receptor_X = train_X_receptor[i: i + BATCH_SIZE].view(-1, 1, RECEPTOR_DIM, RECEPTOR_DIM)
            batch_ligand_X = train_X_ligand[i: i + BATCH_SIZE].view(-1, LIGAND_DIM * LIGAND_DIM)
            batch_y = train_y[i: i + BATCH_SIZE]
            
            batch_receptor_X, batch_ligand_X, batch_y = batch_receptor_X.to(device), batch_ligand_X.to(device), batch_y.to(device)
            
            net.zero_grad()
            
            outputs = net(batch_receptor_X, batch_ligand_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        print("Epoch: ", epoch, "Loss: ", round(float(loss), 3))
        losses.append(loss)
    
    print("Loss value: ", round(float(loss), 3))
    plt.plot(losses)

# Accuracy test
def test_1(net):
    global RECEPTOR_DIM, LIGAND_DIM
    print("begin the test")
    correct = 0
    total = 0

    with torch.no_grad():
        for i in tqdm(range(len(test_X_receptor))):
            real_class = torch.argmax(test_y[i]).to(device)
            receptor = test_X_receptor[i].view(-1, 1, RECEPTOR_DIM, RECEPTOR_DIM)
            ligand = test_X_ligand[i].view(-1, LIGAND_DIM * LIGAND_DIM)
            
            receptor, ligand = receptor.to(device), ligand.to(device)

            net_out = net(receptor, ligand)[0]

            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    print("Accuracy on test_1 samples: ", round(correct / total, 3))
    
   
train(net)
test_1(net)
