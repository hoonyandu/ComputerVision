# model
import torch
import torch.nn as nn
import torch.nn.functional as F

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import time
import copy

from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix

class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()
        self.m_Sigmoid = nn.Sigmoid()

    def forward(self, _x):
        return _x * self.m_Sigmoid(_x)

class SEBlock(nn.Module):
    def __init__(self, _inChannels, _r=4):
        super(SEBlock, self).__init__()

        self.m_Squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.m_Excitation = nn.Sequential(
                                nn.Linear(_inChannels, _inChannels * _r),
                                Swish(),
                                nn.Linear(_inChannels *_r, _inChannels),
                                nn.Sigmoid())

    def forward(self, _x):
        _x = self.m_Squeeze(_x)
        _x = _x.view(_x.size(0), -1)
        _x = self.m_Excitation(_x)
        _x = _x.view(_x.size(0), _x.size(1), 1, 1)

        return _x

class MBConv(nn.Module):
    expand = 6

    def __init__(self, _inChannels, _outChannels, _kernelSize, _stride=1, _seScale=4, _p=0.5):
        super(MBConv, self).__init__()

        # first MBConv is not using stochastic depth
        self.m_P = torch.tensor(_p).float() if (_inChannels == _outChannels) else torch.tensor(1).float()

        self.m_Residual = nn.Sequential(
                            nn.Conv2d(_inChannels, _inChannels * MBConv.expand, 1, stride=_stride, padding=0, bias=False),
                            nn.BatchNorm2d(_inChannels * MBConv.expand, momentum=0.99, eps=1e-3),
                            Swish(),
                            nn.Conv2d(_inChannels * MBConv.expand, _inChannels * MBConv.expand, kernel_size=_kernelSize,
                                     stride=1, padding=_kernelSize//2, bias=False, groups=_inChannels * MBConv.expand),
                            nn.BatchNorm2d(_inChannels * MBConv.expand, momentum=0.99, eps=1e-3),
                            Swish())

        self.m_SE = SEBlock(_inChannels * MBConv.expand, _seScale)

        self.m_Project = nn.Sequential(
                            nn.Conv2d(_inChannels * MBConv.expand, _outChannels, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(_outChannels, momentum=0.99, eps=1e-3))
        self.m_Shortcut = (_stride == 1) and (_inChannels == _outChannels)

    def forward(self, _x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.m_P):
                return _x

        xShortcut = _x
        xResidual = self.m_Residual(_x)
        xSE = self.m_SE(xResidual)

        x = xSE * xResidual
        x = self.m_Project(x)

        if self.m_Shortcut:
            x = xShortcut + x

        return x

class SepConv(nn.Module):
    expand = 1

    def __init__(self, _inChannels, _outChannels, _kernelSize, _stride=1, _seScale=4, _p=0.5):
        super(SepConv, self).__init__()

        # first SepConv is not using stochastic depth
        self.m_P = torch.tensor(_p).float() if (_inChannels == _outChannels) else torch.tensor(1).float()

        self.m_Residual = nn.Sequential(
                            nn.Conv2d(_inChannels * SepConv.expand, _inChannels * SepConv.expand, kernel_size=_kernelSize,
                                    stride=1, padding=_kernelSize//2, bias=False, groups=_inChannels * SepConv.expand),
                            nn.BatchNorm2d(_inChannels * SepConv.expand, momentum=0.99, eps=1e-3),
                            Swish())

        self.m_SE = SEBlock(_inChannels * SepConv.expand, _seScale)

        self.m_Project = nn.Sequential(
                            nn.Conv2d(_inChannels * SepConv.expand, _outChannels, kernel_size=1, stride=1, padding=0, bias=False),
                            nn.BatchNorm2d(_outChannels, momentum=0.99, eps=1e-3))

        self.m_Shortcut = (_stride == 1) and (_inChannels == _outChannels)

    def forward(self, _x):
        # stochastic depth
        if self.training:
            if not torch.bernoulli(self.m_P):
                return _x
        
        xShortcut = _x
        xResidual = self.m_Residual(_x)
        xSE = self.m_SE(xResidual)

        x = xSE * xResidual
        x = self.m_Project(x)

        if self.m_Shortcut:
            x = xShortcut + x

        return x

class EfficientNet(nn.Module):

    def __init__(self, _numClasses=2, _widthCoef=1., _depthCoef=1., _scale=1., _dropout=0.2, _seScale=4, _stochasticDepth=False, _p=0.5):
        super(EfficientNet, self).__init__()
        channels = [32, 16, 24, 40, 80, 112, 192, 320, 1280]
        repeats = [1, 2, 2, 3, 3, 4, 1]
        strides = [1, 2, 2, 2, 1, 2, 1]
        kernelSizes = [3, 3, 5, 3, 5, 5, 3]
        depth = _depthCoef
        width = _widthCoef

        channels = [int(x * width) for x in channels]
        repeats = [int(x * depth) for x in repeats]

        # stochastic depth
        if _stochasticDepth:
            self.m_P = _p
            self.m_Step = (1 - 0.5) / (sum(repeats) - 1)
        else:
            self.m_P = 1
            self.m_Step = 0

        self.m_Upsample = nn.Upsample(scale_factor=_scale, mode='bilinear', align_corners=False)

        self.m_Stage1 = nn.Sequential(nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
                            nn.BatchNorm2d(channels[0], momentum=0.99, eps=1e-3))
        self.m_Stage2 = self.MakeBlock(SepConv, repeats[0], channels[0], channels[1], kernelSizes[0], strides[0], _seScale)
        self.m_Stage3 = self.MakeBlock(MBConv, repeats[1], channels[1], channels[2], kernelSizes[1], strides[1], _seScale)
        self.m_Stage4 = self.MakeBlock(MBConv, repeats[2], channels[2], channels[3], kernelSizes[2], strides[2], _seScale)
        self.m_Stage5 = self.MakeBlock(MBConv, repeats[3], channels[3], channels[4], kernelSizes[3], strides[3], _seScale)
        self.m_Stage6 = self.MakeBlock(MBConv, repeats[4], channels[4], channels[5], kernelSizes[4], strides[4], _seScale)
        self.m_Stage7 = self.MakeBlock(MBConv, repeats[5], channels[5], channels[6], kernelSizes[5], strides[5], _seScale)
        self.m_Stage8 = self.MakeBlock(MBConv, repeats[6], channels[6], channels[7], kernelSizes[6], strides[6], _seScale)
        self.m_Stage9 = nn.Sequential(
                            nn.Conv2d(channels[7], channels[8], 1, stride=1, bias=False),
                            nn.BatchNorm2d(channels[8], momentum=0.99, eps=1e-3),
                            Swish())
        
        self.m_AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.m_Dropout = nn.Dropout(p=_dropout)
        self.m_Linear = nn.Linear(channels[8], _numClasses)

    def forward(self, _x):
        x = self.m_Upsample(_x)
        x = self.m_Stage1(x)
        x = self.m_Stage2(x)
        x = self.m_Stage3(x)
        x = self.m_Stage4(x)
        x = self.m_Stage5(x)
        x = self.m_Stage6(x)
        x = self.m_Stage7(x)
        x = self.m_Stage8(x)
        x = self.m_Stage9(x)
        x = self.m_AvgPool(x)
        x = x.view(x.size(0), -1)
        x = self.m_Dropout(x)
        x = self.m_Linear(x)
        
        return x

    def MakeBlock(self, _block, _repeats, _inChannels, _outChannels, _kernelSize, _stride, _seScale):
        strides = [_stride] + [1] * (_repeats - 1)
        layers = list()
        
        for stride in strides:
            layers.append(_block(_inChannels, _outChannels, _kernelSize, stride, _seScale, self.m_P))
            _inChannels = _outChannels
            self.m_P = self.m_P - self.m_Step
            
        return nn.Sequential(*layers)

class EfficientNetBlock():

    def __init__(self):
        pass

    def EfficientNetB0(self, _numClasses=2):
        return EfficientNet(_numClasses=_numClasses, _widthCoef=1.0, _depthCoef=1.0, _scale=1.0, _dropout=0.2, _seScale=4)

    def EfficientNetB1(self, _numClasses=2):
        return EfficientNet(_numClasses=_numClasses, _widthCoef=1.0, _depthCoef=1.1, _scale=240/224, _dropout=0.2, _seScale=4)

    def EfficientNetB2(self, _numClasses=2):
        return EfficientNet(_numClasses=_numClasses, _widthCoef=1.1, _depthCoef=1.2, _scale=260/224, _dropout=0.3, _seScale=4)

    def EfficientNetB3(self, _numClasses=2):
        return EfficientNet(_numClasses=_numClasses, _widthCoef=1.2, _depthCoef=1.4, _scale=300/224, _dropout=0.3, _seScale=4)

    def EfficientNetB4(self, _numClasses=2):
        return EfficientNet(_numClasses=_numClasses, _widthCoef=1.4, _depthCoef=1.8 , _scale=380/224, _dropout=0.4, _seScale=4)

    def EfficientNetB5(self, _numClasses=2):
        return EfficientNet(_numClasses=_numClasses, _widthCoef=1.6, _depthCoef=2.2, _scale=456/224, _dropout=0.4, _seScale=4)

    def EfficientNetB6(self, _numClasses=2):
        return EfficientNet(_numClasses=_numClasses, _widthCoef=1.8, _depthCoef=2.6, _scale=528/224, _dropout=0.5, _seScale=4)

    def EfficientNetB7(self, _numClasses=2):
        return EfficientNet(_numClasses=_numClasses, _widthCoef=2.0, _depthCoef=3.1, _scale=600/224, _dropout=0.5, _seScale=4)


class MakeModel():
    def __init__(self):
        pass

    def GetLR(self, _opt):
        # get current lr
        for paramGroup in _opt.param_groups:
            return paramGroup['lr']

    def MetricBatch(self, _output, _target):
        # calculate the metric per mini-batch
        pred = _output.argmax(1, keepdim=True)
        corrects = pred.eq(_target.view_as(pred)).sum().item()
        return corrects

    def LossBatch(self, _lossFunc, _output, _target, _opt=None):
        # calculate the loss per mini-batch
        lossB = _lossFunc(_output, _target)
        metricB = self.MetricBatch(_output, _target)

        if _opt is not None:
            _opt.zero_grad()
            lossB.backward()
            _opt.step()

        return lossB.item(), metricB

    def LossEpoch(self, _device, _model, _lossFunc, _dataSetLoader, _sanityCheck=False, _opt=None, _testCheck=False):
        runningLoss = 0.0
        runningMetric = 0.0
        lenData = len(_dataSetLoader.dataset)

        for xb, yb in _dataSetLoader:
            xb = xb.to(_device)
            yb = yb.to(_device)
            output = _model(xb)

            lossB, metricB = self.LossBatch(_lossFunc, output, yb, _opt)

            runningLoss += lossB

            if metricB is not None:
                runningMetric += metricB
            if _sanityCheck is True:
                break

            if _testCheck is True:
                _, predicted = output.data.max(1)
                labels = yb.tolist()
                predicted = predicted.tolist()        

        loss = runningLoss / lenData
        metric = runningMetric / lenData
        return loss, metric

    def LossEpochTest(self, _device, _model, _lossFunc, _dataSetLoader, _sanityCheck=False):
        runningLoss = 0.0
        runningMetric = 0.0
        lenData = len(_dataSetLoader.dataset)
        labelList = list()
        predictedList = list()

        for xb, yb in _dataSetLoader:
            xb = xb.to(_device)
            yb = yb.to(_device)
            output = _model(xb)

            lossB, metricB = self.LossBatch(_lossFunc, output, yb)

            runningLoss += lossB

            _, predicted = output.data.max(1)
            labelList.extend(yb.tolist())
            predictedList.extend(predicted.tolist())

            if metricB is not None:
                runningMetric += metricB
            if _sanityCheck is True:
                break     

        loss = runningLoss / lenData
        metric = runningMetric / lenData
        return loss, metric, labelList, predictedList


    def TrainVal(self, _model, _params, _lr):

        opt = torch.optim.Adam(_model.parameters(), lr= _lr)
        lossFunc = torch.nn.CrossEntropyLoss(reduction='sum')
        lrScheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=10)

        device = _params['device']
        numEpochs = _params['numEpochs']
        trainDataLoader = _params['trainDataLoader']
        validationDataLoader = _params['validationDataLoader']
        sanityCheck = _params['sanityCheck']
        pathToWeights = _params['pathToWeights']

        lossHistory = {'train':list(), 'validation': list()}
        metricHistory = {'train': list(), 'validation': list()}

        bestLoss = float('inf')
        bestModelWeights = copy.deepcopy(_model.state_dict())
        startTime = time.time()

        for epoch in range(numEpochs):
            currentLR = self.GetLR(opt)
            print('Epoch {}/{}, current lr = {}'.format(epoch, numEpochs-1, currentLR))

            _model.train()
            trainLoss, trainMetric = self.LossEpoch(device, _model, lossFunc, trainDataLoader, sanityCheck, opt)
            lossHistory['train'].append(trainLoss)
            metricHistory['train'].append(trainMetric)

            _model.eval()
            with torch.no_grad():
                valLoss, valMetric = self.LossEpoch(device, _model, lossFunc, validationDataLoader, sanityCheck)
                lossHistory['validation'].append(valLoss)
                metricHistory['validation'].append(valMetric)

            if valLoss < bestLoss:
                bestLoss = valLoss
                bestModelWeights = copy.deepcopy(_model.state_dict())
                torch.save(_model.state_dict(), pathToWeights)
                print('Copied best model weights!')

            lrScheduler.step(valLoss)
            if currentLR != self.GetLR(opt):
                print("Loading best model weights!")
                _model.load_state_dict(bestModelWeights)

            print('train loss: %.6f, val loss: %.6f, accuracy: %.2f, time: %.4f min' %
                (trainLoss, valLoss, 100 * valMetric, (time.time() - startTime) / 60))

            print('-' * 15)

        _model.load_state_dict(bestModelWeights)
        return _model, lossHistory, metricHistory

    def Testing(self, _model, _params):
        
        torch.manual_seed(0)

        device = _params['device']
        testDataLoader = _params['testDataLoader']
        sanityCheck = _params['sanityCheck']
        bLoad = _params['bLoad']
        loadPath = _params['loadPath']

        lossFunc = torch.nn.CrossEntropyLoss(reduction="sum")
        startTime = time.time()
        
        if bLoad == True:
            _model.load_state_dict(torch.load(loadPath, map_location=device))
            print("model loaded [", loadPath, "]")
        else:
            model = _model

        _model.eval()
        with torch.no_grad():
            testLoss, testMetric, labelList, predList = self.LossEpochTest(device, _model, lossFunc, testDataLoader, sanityCheck)

        print('test loss: %.6f, accuracy: %.2f, time: %.4f min' %
        (testLoss, 100 * testMetric, (time.time() - startTime) / 60))
        print('-' * 15)

        return labelList, predList

    def PlotGraph(self, _numEpochs, _lossHistory, _metricHistory, _savePath):

        fig, ax = plt.subplots(1, 2)
        ax[0, 0].plot(range(1, _numEpochs+1), _lossHistory['train'], label='train')
        ax[0, 0].plot(range(1, _numEpochs+1), _lossHistory['validation'], label='validation')
        ax[0, 0].set_title('Train-Val Loss')
        ax[0, 0].set_xlabel('Training Epochs')
        ax[0, 0].set_ylabel('Loss')

        ax[0, 1].plot(range(1, _numEpochs+1), _metricHistory['train'], label='train')
        ax[0, 1].plot(range(1, _numEpochs+1), _metricHistory['validation'], label='validation')
        ax[0, 1].set_title('Train-Val Accuracy')
        ax[0, 1].set_xlabel('Training Epochs')
        ax[0, 1].set_ylabel('Accuracy')

        fig.tight_layout()
        plt.show()
        plt.savefig(_savePath + 'train-val_loss_acc.png', format='png', dpi=300)

    def ConfusionMatricSklearn(self, _numClasses, _label,_predicted):
        labels = _numClasses # malware, benigh
        cm = confusion_matrix(_label, _predicted, labels=[i for i in range(labels)])
        print(cm)
