
import time
import torch
import torch.nn as nn
from torchsummary import summary

import sys

sys.path.append('/data/bfb/CMD')

from torch.utils.data import DataLoader

from loadECG.DataSet import ECGDataSet
from trainF.Test_evaluate import test_evaluate


class BasicBlock1d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock1d, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=7, stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.2)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
        
    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device) # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class Model(nn.Module):
    def __init__(self, block, layers, input_channels=12, inplanes=64, num_classes=6):
        super(Model, self).__init__()
        self.inplanes = inplanes
        self.conv1 = nn.Conv1d(input_channels, self.inplanes, kernel_size=15, stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock1d, 64, layers[0])
        self.layer2 = self._make_layer(BasicBlock1d, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(BasicBlock1d, 256, layers[2], stride=2)
        self.adaptiveavgpool = nn.AdaptiveAvgPool1d(1)
        self.adaptivemaxpool = nn.AdaptiveMaxPool1d(1)
        # self.fc = nn.Linear(512 * block.expansion * 2, num_classes)
        self.dropout = nn.Dropout(0.2)

        self.bilstm = nn.LSTM(input_size=19, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.fc_ecg = nn.Linear(256, 16)  # 2 for bidirection
        self.fc_mtf = nn.Linear(132, 8)
        self.fc_expert = nn.Linear(2, 8)
        self.fc = nn.Linear(32,2)
   
    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x1, x2, x3, x4):
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.maxpool(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1,_ = self.bilstm(x1)
        x1 = self.fc_ecg(x1[:,-1,:])
        x2 = torch.flatten(x2,1)
        x2 = self.fc_mtf(x2)
        ecg_features = torch.cat((x1,x2),dim=1)
        x3 = x3.reshape(x3.shape[0],1)
        x4 = x4.reshape(x4.shape[0],1)
        expert_features = torch.cat((x3,x4),dim=1)
        expert_features = self.fc_expert(expert_features)
        features = torch.cat((ecg_features,expert_features),dim=1)
        features = self.fc(features)
        return features




def Fusion_Model(**kwargs):
    model = Model(BasicBlock1d, [1, 1, 1], **kwargs)
    return model


if __name__ == '__main__':
    filename = r'../DATA/DATA_Test01.pkl'

    print("Data loaded successfully!")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batchsz = 1
    # 读取数据
    test_db = ECGDataSet(filename, mode='test')
    print(test_db)

    test_loader = DataLoader(test_db, batch_size=batchsz, shuffle=True)
    print(test_loader)
    
    model = Fusion_Model()
    for step, (x1, x2, x3, x4, y) in enumerate(test_loader):
            # x1: torch.Size([batchsz, 12, 300])   x2:torch.Size([batchsz, 12, 11])
            output = model(x1, x2, x3, x4).to(device)
    print(output.shape)
    print(model)
    # summary(model=model, input_size=(12, 300), device='cpu')