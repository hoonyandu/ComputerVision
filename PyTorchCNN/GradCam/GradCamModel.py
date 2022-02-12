import torch
import torch.nn as nn
from torch.autograd import Variable
import pkbar

import cv2
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, _classNum = 2, _featureLayer = 8):
        super(CNN, self).__init__()
        self.m_ClassNum = _classNum
        self.m_FeatureLayer = _featureLayer
        self.m_GradType = 'required'
        self.m_Gradients = None

        conv1OutChannel = 32
        conv2OutChannel = 64
        conv3OutChannel = 64
        conv1KernelSize = 3
        conv2KernelSize = 2
        conv3KernelSize = 2
        poolSize = 2

        self.m_Layer = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = conv1OutChannel, kernel_size = conv1KernelSize, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(poolSize, poolSize),

            nn.Conv2d(in_channels = conv1OutChannel, out_channels = conv2OutChannel, kernel_size = conv2KernelSize, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(poolSize, poolSize),

            nn.Conv2d(in_channels = conv2OutChannel, out_channels = conv3OutChannel, kernel_size = conv3KernelSize),
            nn.ReLU(),
            nn.MaxPool2d(poolSize, poolSize),
        )

        self.m_FCLayer = nn.Sequential(
            nn.Linear(36864, 256),
            nn.ReLU(),
            nn.Dropout2d(0.5),

            nn.Linear(256, _classNum)
        )

    def ActivationHook(self, _grad):
        self.m_Gradients = _grad

    def forward(self, _x):
        
        if self.m_GradType != 'required':
            out = self.m_Layer(_x)
            out = out.view(_x.size()[0], -1)
            out = self.m_FCLayer(out)

        elif self.m_GradType == 'required':
            hook = self.m_Layer[:self.m_FeatureLayer](_x)
            hook.register_hook(self.ActivationHook)
            self.m_Gradients = hook

            out = self.m_Layer(_x)
            out = out.view(_x.size()[0], -1)
            out = self.m_FCLayer(out)

        return out

    def GetActivationsGradient(self):
        return self.m_Gradients

    def GetActivations(self, _x):
        return self.m_Layer[:self.m_FeatureLayer](_x)


class MakeModel():

    def __init__(self):
        pass

    def GradCamTraining(self, _device, _model, _epoch, _lr, _trainData, _valData, _testData, _testSetList, _imgPath, _bSave = False, _savePath = None):
        Image.MAX_IMAGE_PIXELS = 933120000

        # _model.train()

        optimizer = torch.optim.Adam(_model.parameters(), lr = _lr)
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(_epoch):
            _model.train()

            trainLoss = 0.0
            trainSize = 0.0
            trainCorrect = 0.0

            print("Epoch {}/{}".format(epoch + 1, _epoch))
            progress = pkbar.Kbar(target=len(_trainData), width = 25)

            # train 
            for batchIdx, data in enumerate(_trainData):
                images, labels = data
                images = Variable(images).to(_device)
                labels = Variable(labels).to(_device)
                
                optimizer.zero_grad()
                outputs = _model.forward(images)

                loss = criterion(outputs, labels)
                loss.backward()
                #trainRMSE = torch.sqrt(trainLoss)
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
            _model.m_GradType = 'no_grad'
            with torch.no_grad():
                valLoss = 0.0
                valSize = 0.0
                valCorrect = 0.0

                for batchIdx, data in enumerate(_valData):
                    images, labels = data
                    images = Variable(images).to(_device)
                    labels = Variable(labels).to(_device)
                    
                    outputs = _model.forward(images)
                    valLoss = criterion(outputs, labels).item()

                    _, predicted = outputs.max(1)
                    valSize += labels.shape[0]

                    valCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                    valAccuracy = 100 * valCorrect / valSize

                progress.add(1, values=[("val loss", valLoss), ("val acc", valAccuracy)])

            # testing
            _model.m_GradType = 'required'
            progressTest = pkbar.Kbar(target=len(_testData), width = 25)
            # with torch.no_grad():
            testLoss = 0.0
            testSize = 0.0
            testCorrect = 0.0

            for batchIdx, data in enumerate(_testData):
                images, labels = data
                images = Variable(images, volatile=True).to(_device)
                labels = Variable(labels).to(_device)
                outputs = _model.forward(images)

                # values, idx = output.max(dim=1)
                gradients = _model.GetActivationsGradient()
                pooledGradients = torch.mean(gradients, dim=[0,2,3])
                featureLayer = _model.GetActivations(images).detach()

                for i in range(64):
                    featureLayer[:, i, :, :] *= pooledGradients[i]

                heatmap = torch.mean(featureLayer, dim=1).squeeze()
                heatmap = np.maximum(heatmap.detach(), 0)
                heatmap /= torch.max(heatmap)

                filePath = _imgPath + _testSetList[batchIdx][0].split('/')[-2] + '/' + _testSetList[batchIdx][0].split('/')[-1][:-4] + '_epoch' + str(epoch + 1) + '.png'
                # plt.savefig(filePath, dpi=200)
                plt.imsave(filePath, heatmap.squeeze())

                testLoss = criterion(outputs, labels).item()

                _, predicted = outputs.data.max(1)
                testSize += labels.shape[0]
                # testCorrect += predicted.eq(labels.data).sum().item()
                testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                accuracy = 100 * testCorrect / testSize
                
                progressTest.update(batchIdx, values = [("test loss: ", testLoss), ("test acc: ", accuracy)])

            testLoss /= len(_testData.dataset)
        progressTest.add(1)

        if _bSave == True:
            torch.save(_model, _savePath)
            print("model saved [", _savePath, "]")

    def GradCamTest(self, _device, _model, _lr, _testData, _testSetList, _imgPath, _bLoad = False, _loadPath = None):
        Image.MAX_IMAGE_PIXELS = 933120000

        optimizer = torch.optim.Adam(_model.parameters(), lr = _lr)
        criterion = torch.nn.CrossEntropyLoss()

        torch.manual_seed(0)
        if _bLoad == True:
            model = torch.load(_loadPath, map_location=_device)
            print("model loaded [", _loadPath, "]")
        else:
            model = _model

        model.eval()

        testLoss = 0.0
        testSize = 0.0
        testCorrect = 0.0

        progressTest = pkbar.Kbar(target=len(_testData), width = 25)

        for batchIdx, data in enumerate(_testData):
            images, labels = data
            images = Variable(images, volatile=True).to(_device)
            labels = Variable(labels).to(_device)
            outputs = model.forward(images)

            # values, idx = output.max(dim=1)
            gradients = model.GetActivationsGradient()
            pooledGradients = torch.mean(gradients, dim=[0,2,3])
            featureLayer = model.GetActivations(images).detach()

            for i in range(64):
                featureLayer[:, i, :, :] *= pooledGradients[i]

            heatmap = torch.mean(featureLayer, dim=1).squeeze()
            heatmap = np.maximum(heatmap.detach(), 0)
            heatmap /= torch.max(heatmap)

            filePath = _imgPath + _testSetList[batchIdx][0].split('/')[-2] + '/' + _testSetList[batchIdx][0].split('/')[-1]
            # plt.savefig(filePath, dpi=200)
            plt.imsave(filePath, heatmap.squeeze())

            testLoss = criterion(outputs, labels).item()

            _, predicted = outputs.data.max(1)
            testSize += labels.shape[0]
            # testCorrect += predicted.eq(labels.data).sum().item()
            testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
            accuracy = 100 * testCorrect / testSize
            
            progressTest.update(batchIdx, values = [("test loss: ", testLoss), ("test acc: ", accuracy)])

        testLoss /= len(_testData.dataset)
        progressTest.add(1)


    def Training(self, _device, _model, _epoch, _lr, _trainData, _valData, _bSave = False, _savePath = None):
        # DecompressionBombWarning issue 210412
        # https://github.com/zimeon/iiif/issues/11
        # Image.warnings.simplefilter('ignore', Image.DecompressionBombWarning)

        # https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
        Image.MAX_IMAGE_PIXELS = 933120000

        _model.train()

        optimizer = torch.optim.Adam(_model.parameters(), lr = _lr)
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
                images = Variable(images).to(_device)
                labels = Variable(labels).to(_device)
                
                optimizer.zero_grad()
                outputs = _model(images)

                loss = criterion(outputs, labels)
                loss.backward()
                #trainRMSE = torch.sqrt(trainLoss)
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
                    images = Variable(images).to(_device)
                    labels = Variable(labels).to(_device)
                    
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
        # https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
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
                images = Variable(images).to(_device)
                labels = Variable(labels).to(_device)
                outputs = model(images)

                testLoss = criterion(outputs, labels).item()

                _, predicted = outputs.data.max(1)
                testSize += labels.shape[0]
                # testCorrect += predicted.eq(labels.data).sum().item()
                testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
                accuracy = 100 * testCorrect / testSize
                
                progress.update(batchIdx, values = [("test loss: ", testLoss), ("test acc: ", accuracy)])
            
            testLoss /= len(_testData.dataset)
        progress.add(1)

    def TestingAccuracy(self, _device, _model, _lr, _testData, _bLoad = False, _loadPath = None):
        # https://stackoverflow.com/questions/51152059/pillow-in-python-wont-let-me-open-image-exceeds-limit
        Image.MAX_IMAGE_PIXELS = 933120000

        labelList = list()
        predictedList = list()

        optimizer = torch.optim.Adam(_model.parameters(), lr = _lr)
        criterion = torch.nn.CrossEntropyLoss()

        torch.manual_seed(0)
        if _bLoad == True:
            # 2021-03-04 HOONY
            model = torch.load(_loadPath, map_location=_device)
            print("model loaded [", _loadPath, "]")
        else:
            model = _model

        model.eval()

        testLoss = 0.0
        testSize = 0.0
        testCorrect = 0.0

        progress = pkbar.Kbar(target=len(_testData), width = 25)

        for batchIdx, data in enumerate(_testData):
            images, labels = data
            # images, labels = images.to(_device), labels.to(_device)
            images = Variable(images, volatile=True).to(_device)
            labels = Variable(labels).to(_device)
            outputs = model(images)

            testLoss = criterion(outputs, labels).item()

            _, predicted = outputs.data.max(1)
            testSize += labels.shape[0]
            # testCorrect += predicted.eq(labels.data).sum().item()
            labelList.extend(labels.tolist())
            predictedList.extend(predicted.tolist())
            testCorrect += predicted.eq(labels.view_as(predicted)).sum().item()
            accuracy = 100 * testCorrect / testSize
            
            progress.update(batchIdx, values = [("test loss: ", testLoss), ("test acc: ", accuracy)])
        
        testLoss /= len(_testData.dataset)
        progress.add(1)

        return labelList, predictedList