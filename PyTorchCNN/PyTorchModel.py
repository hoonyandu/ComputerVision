import torch
import torch.nn as nn
import pkbar

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import itertools
import numpy as np

from PIL import Image

class PillModel(nn.Module):

    def __init__(self, _classNum):
        super(PillModel, self).__init__()

        self.m_ClassNum = _classNum

        conv1OutChannel = 32
        conv2OutChannel = 64
        conv3OutChannel = 64
        conv1KernelSize = 3
        conv2KernelSize = 2
        conv3KernelSize = 2
        poolSize = 2

        self.m_Conv1 = nn.Conv2d(
            in_channels = 3,
            out_channels = conv1OutChannel,
            kernel_size = conv1KernelSize,
            padding = 1)
        self.m_Pool1 = nn.MaxPool2d(poolSize, poolSize)

        self.m_Conv2 = nn.Conv2d(
            in_channels = conv1OutChannel,
            out_channels = conv2OutChannel,
            kernel_size = conv2KernelSize,
            padding = 1)
        self.m_Pool2 = nn.MaxPool2d(poolSize, poolSize)

        self.m_Conv3 = nn.Conv2d(
            in_channels = conv2OutChannel,
            out_channels = conv3OutChannel,
            kernel_size = conv3KernelSize)
        self.m_Pool3 = nn.MaxPool2d(poolSize, poolSize)

        self.m_Linear4 = nn.Linear(46656, 256)
        self.m_Drop4 = nn.Dropout2d(0.5)

        self.m_Linear5 = nn.Linear(256, _classNum)
        self.m_Relu = nn.ReLU()
        self.m_Softmax = nn.Softmax()

    def forward(self, x):

        x = self.m_Relu(self.m_Conv1(x))
        x = self.m_Pool1(x)

        x = self.m_Relu(self.m_Conv2(x))
        x = self.m_Pool2(x)

        x = self.m_Relu(self.m_Conv3(x))
        x = self.m_Pool3(x)

        x = x.view(x.shape[0],-1)
        x = self.m_Relu(self.m_Linear4(x))
        x = self.m_Drop4(x)

        x = self.m_Linear5(x)
        
        return x

import torch.optim as optim


class MakeModel():

    def __init__(self):
        pass

    def Training(self, _device, _model, _epoch, _lr, _trainData, _valData, _bSave = False, _savePath = None):

        Image.MAX_IMAGE_PIXELS = 933120000

        _model.train()

        optimizer = optim.Adam(_model.parameters(), lr = _lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        for epoch in range(_epoch):
            trainLoss = 0.0
            trainSize = 0.0
            trainCorrect = 0.0

            print("Epoch {}/{}".format(epoch + 1, _epoch))
            progress = pkbar.Kbar(target=len(_trainData), width = 25)

            # train 
            for batchIdx, data in enumerate(_trainData):
                images, labels = data
                images, labels = images.to(_device), labels.to(_device)
                
                optimizer.zero_grad()
                outputs = _model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                trainLoss = loss.item()

                _, predicted = outputs.max(1)
                trainSize += labels.shape[0]
                trainCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                trainAccuracy = 100 * trainCorrect / trainSize

                progress.update(batchIdx, values = [("loss: ", trainLoss), ("acc: ", trainAccuracy)])

                del loss
                del outputs

            # validation
            with torch.no_grad():
                valLoss = 0.0
                valSize = 0.0
                valCorrect = 0.0

                for batchIdx, data in enumerate(_valData):
                    images, labels = data
                    images, labels = images.to(_device), labels.to(_device)
                    
                    outputs = _model(images)
                    valLoss = criterion(outputs, labels).item()

                    _, predicted = outputs.max(1)
                    valSize += labels.shape[0]

                    valCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                    valAccuracy = 100 * valCorrect / valSize

                progress.add(1, values=[("val loss", valLoss), ("val acc", valAccuracy)])

        if _bSave == True:
            torch.save(_model, _savePath)
            print("model saved [", _savePath, "]")

    def Testing(self, _device, _model, _testData, _bLoad = False, _loadPath = None):
        Image.MAX_IMAGE_PIXELS = 933120000

        torch.manual_seed(0)
        if _bLoad == True:
            # 2021-03-04 HOONY
            model = torch.load(_loadPath, map_location=_device)
            print("model loaded [", _loadPath, "]")
        else:
            model = _model

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        testLoss = 0.0
        testSize = 0.0
        testCorrect = 0.0

        progress = pkbar.Kbar(target=len(_testData), width = 25)

        with torch.no_grad():
            for batchIdx, data in enumerate(_testData):
                images, labels = data
                images, labels = images.to(_device), labels.to(_device)
                outputs = model(images)

                testLoss = criterion(outputs, labels).item()

                _, predicted = outputs.data.max(1)
                testSize += labels.shape[0]
                testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                accuracy = 100 * testCorrect / testSize
                
                progress.update(batchIdx, values = [("test loss: ", testLoss), ("test acc: ", accuracy)])
            
            testLoss /= len(_testData.dataset)
        progress.add(1)

    def TestingAccuracy(self, _device, _model, _testData, _bLoad = False, _loadPath = None):
        Image.MAX_IMAGE_PIXELS = 933120000

        labelList = list()
        predictedList = list()

        torch.manual_seed(0)
        if _bLoad == True:
            model = torch.load(_loadPath, map_location=_device)
            print("model loaded [", _loadPath, "]")
        else:
            model = _model

        model.eval()
        criterion = torch.nn.CrossEntropyLoss()

        testLoss = 0.0
        testSize = 0.0
        testCorrect = 0.0

        progress = pkbar.Kbar(target=len(_testData), width = 25)

        with torch.no_grad():
            for batchIdx, data in enumerate(_testData):
                images, labels = data
                images, labels = images.to(_device), labels.to(_device)
                outputs = model(images)

                testLoss = criterion(outputs, labels).item()

                _, predicted = outputs.data.max(1)
                testSize += labels.shape[0]
                labelList.extend(labels.tolist())
                predictedList.extend(predicted.tolist())
                testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                accuracy = 100 * testCorrect / testSize
                
                progress.update(batchIdx, values = [("test loss: ", testLoss), ("test acc: ", accuracy)])
            
            testLoss /= len(_testData.dataset)

        progress.add(1)

        return labelList, predictedList

    def ConfusionMatrixSklearn(self, _label, _predicted):
        labels = 2 # malware, benign
        cm = confusion_matrix(_label, _predicted, labels=[i for i in range(labels)])
        print(cm)
