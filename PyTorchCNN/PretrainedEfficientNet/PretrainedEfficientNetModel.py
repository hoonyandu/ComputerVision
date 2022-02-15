from efficientnet_pytorch import EfficientNet
from sklearn.metrics import confusion_matrix

import numpy as np
import json
from PIL import Image
import torch
import matplotlib.pyplot as plt
import time
import os
import copy
import random


# https://keep-steady.tistory.com/35
class PretrainedEfficientNetModel():
    def __init__(self):
        pass

    def TrainModel(self, _model, _params, _numEpochs=25):
        start = time.time()
        bestModelWegiths = copy.deepcopy(_model.state_dict())
        bestAcc = 0.0
        trainLoss, trainAcc, validLoss, validAcc = list(), list(), list(), list()
        dataloaders = dict()

        device = _params['device']
        criterion = _params['criterion']
        optimizer = _params['optimizer']
        scheduler = _params['scheduler']
        numEpochs = _params['numEpochs']
        trainDataLoader = _params['trainDataLoader']
        validationDataLoader = _params['validationDataLoader']
        pathToWeights = _params['pathToWeights']
        model = _model

        dataloaders['train'] = trainDataLoader
        dataloaders['valid'] = validationDataLoader

        for epoch in range(numEpochs):
            print('Epoch {}/{}'.format(epoch, _numEpochs-1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'valid']:
                if phase == 'train':
                    # Set model to training mode
                    model.train()
                else:
                    # Set model to evaluate mode
                    model.eval()

                runningLoss, runningCorrects, numCnt = 0.0, 0, 0

                # Iterate over data
                for inputs, labels in dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    # zero the parameter gradients
                    optimizer.zero_grad()

                    # forward
                    # track history if only in train

                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # backward + optimizer only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    # statistics
                    runningLoss += loss.item() * inputs.size(0)
                    runningCorrects += torch.sum(preds == labels.data)
                    numCnt += len(labels)

                if phase == 'train':
                    scheduler.step()

                epochLoss = float(runningLoss / numCnt)
                epochAcc = float((runningCorrects.double() / numCnt).cpu()*100)

                if phase == 'train':
                    trainLoss.append(epochLoss)
                    trainAcc.append(epochAcc)

                else:
                    validLoss.append(epochLoss)
                    validAcc.append(epochAcc)
                
                print('{} Loss: {:.2f} Acc: {:.1f}'.format(phase, epochLoss, epochAcc))
            
                # deep copy the model
                if phase == 'valid' and epochAcc > bestAcc:
                    bestIdx = epoch
                    bestAcc = epochAcc
                    bestModelWegiths = copy.deepcopy(model.state_dict())
                    # best_model_wts = copy.deepcopy(model.module.state_dict())
                    print('==> best model saved - %d / %.1f'%(bestIdx, bestAcc))

        timeElapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(timeElapsed // 60, timeElapsed % 60))
        print('Best valid Acc: %d - %.1f' % (bestIdx, bestAcc))

        # load best model weights
        model.load_state_dict(bestModelWegiths)
        # torch.save(model.state_dict(), 'president_model.pt')
        torch.save(model.state_dict(), pathToWeights)
        print('model saved')
        return model, bestIdx, bestAcc, trainLoss, trainAcc, validLoss, validAcc

    def Test(self, _model, _params, _numImages=4):

        device = _params['device']
        criterion = _params['criterion']
        testDataLoader = _params['testDataLoader']
        bLoad = _params['bLoad']
        savePath = _params['savePath']
        loadPath = _params['loadPath']

        if bLoad == True:
            _model.load_state_dict(torch.load(loadPath, map_location=device))
            print("model loaded [", loadPath, "]")
            
        _model.eval()
        fig = plt.figure()
        labelList = list()
        predictedList = list()
        
        runningLoss, runningCorrects, numCnt = 0.0, 0, 0

        with torch.no_grad():
            for inputs, labels in testDataLoader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = _model(inputs)
                _, preds = torch.max(outputs, 1)
                labelList.extend(labels.tolist())
                predictedList.extend(preds.tolist())
                loss = criterion(outputs, labels)

                runningLoss += loss.item() * inputs.size(0)
                runningCorrects += torch.sum(preds == labels.data)
                numCnt += inputs.size(0)

            testLoss = runningLoss / numCnt
            testAcc = runningCorrects.double() / numCnt
            print('test done : loss / acc : %.2f / %.1f' % (testLoss, testAcc*100))

        # with torch.no_grad():
        #     for inputs, labels in testDataLoader:
        #         inputs = inputs.to(device)
        #         labels = labels.to(device)

        #         outputs = _model(inputs)
        #         _, preds = torch.max(outputs, 1)

        #         for j in range(1, _numImages+1):
        #             ax = plt.subplot(_numImages//2, 2, j)
        #             ax.axis('off')
        #             ax.set_title('%s : %s -> %s' % (
        #                 'True' if str(labels[j].cpu().numpy()) == str(preds[j].cpu().numpy()) else 'False',
        #                 str(labels[j].cpu().numpy()),
        #                 str(preds[j].cpu().numpy())
        #                 ))
        #             # self.ImgShow(inputs.cpu().data[j])
        #             img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
        #             mean = np.array([0.485, 0.456, 0.406])
        #             std = np.array([0.229, 0.224, 0.225])
        #             img = std * img + mean
        #             img = np.clip(img, 0, 1)
        #             plt.show(img)

        #             plt.savefig(savePath + 'testImage_' + str(j) + '.png', format='png', dpi=300)

        return labelList, predictedList

    def ImgShow(self, _inp, _title=None):
        '''Img Show for Tensor'''
        inp = _inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        # plt.imshow(inp)
        if _title is not None:
            plt.title(_title)

    def PlotGraph(self, _bestIdx, _bestAcc, _trainLoss, _trainAcc, _validLoss, _validAcc, _savePath):
        print('best model : %d - %1.f / %.1f' % (_bestIdx, _validAcc[_bestIdx], _validLoss[_bestIdx]))
        fig, ax1 = plt.subplots()

        ax1.plot(_trainAcc, 'b-', label="train acc")
        ax1.plot(_validAcc, 'r-', label="valid acc")
        plt.plot(_bestIdx, _validAcc[_bestIdx], 'ro')
        ax1.set_xlabel('epoch')
        # Make the y-axis label, ticks and tick labels match the line color
        ax1.set_ylabel('acc', color='k')
        ax1.tick_params('y', colors='k')

        ax2 = ax1.twinx()
        ax2.plot(_trainLoss, 'g-', label="train loss")
        ax2.plot(_validLoss, 'k-', label="valid loss")
        plt.plot(_bestIdx, _validLoss[_bestIdx], 'ro')
        ax2.set_ylabel('loss', color='k')
        ax2.tick_params('y', colors='k')

        plt.legend()

        fig.tight_layout()
        plt.show()
        plt.savefig(_savePath + 'train-val_loss_acc.png', format='png', dpi=300)

    def ConfusionMatricSklearn(self, _numClasses, _label,_predicted):
        labels = _numClasses # malware, benigh
        cm = confusion_matrix(_label, _predicted, labels=[i for i in range(labels)])
        print(cm)