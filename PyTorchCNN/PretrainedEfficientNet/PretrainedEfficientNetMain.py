import torch
from torchsummary import summary

import PretrainedEfficientNetModel
import PyTorchData
from efficientnet_pytorch import EfficientNet

class PretrainedEfficientNetMain():
    def __init__(self):
        self.m_Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device type [", self.m_Device, "]")

        self.m_NumClasses = 2
        self.m_InputDim = 224
        # self.m_BatchSize = 32
        self.m_BatchSize = 16
        # self.m_LearningRate = 0.05
        # self.m_LearningRate = 0.01
        self.m_LearningRate = 0.0004
        self.m_Epochs = 10
        self.m_ModelName = "efficientnet-b5"

        self.m_cPretrainedModel = PretrainedEfficientNetModel.PretrainedEfficientNetModel()
        self.m_cPytorchData = PyTorchData.PyTorchData(
            _dataType = "image",
            _dataPath = '/home/yoon/paper/image/rgbEntire/2019/',
            _inputDim = self.m_InputDim,
            _batchSize = self.m_BatchSize
        )

    def main(self):
        print("\n[ Model ]\n")
        imageSize = EfficientNet.get_image_size(self.m_ModelName)
        print(imageSize)
        model = EfficientNet.from_pretrained(self.m_ModelName, num_classes = self.m_NumClasses)
        print(model)
        summary(model, (3, self.m_InputDim, self.m_InputDim), device = self.m_Device.type)
        optimizer = torch.optim.SGD([
                                    {
                                    'params':model.parameters(),
                                    'lr':self.m_LearningRate,
                                    'momentum':0.9,
                                    'weight_decay':1e-4}])

        print("\n[ Data ]\n")
        trainData = self.m_cPytorchData.ImageTrain()
        valData = self.m_cPytorchData.ImageValidation()
        testData, _ = self.m_cPytorchData.ImageTest()

        print("\n[ Training ]\n")
        # lmbda = lambda epoch: 0.98739**epoch
        lmbda = lambda epoch: 0.98739
        # print(optimizer.param_groups[0])
        paramsTrain = {
            'device': self.m_Device,
            'criterion': torch.nn.CrossEntropyLoss(),
            'optimizer': optimizer,
            'scheduler': torch.optim.lr_scheduler.MultiplicativeLR(optimizer=optimizer, lr_lambda=lmbda),
            'numEpochs': self.m_Epochs,
            'trainDataLoader': trainData,
            'validationDataLoader': valData,
            'sanityCheck': False,
            'pathToWeights': '/home/yoon/paper/source/model/2021-11-30_2019_rgbEntire.preEfficientNetB5.pt'
        }

        paramsTest = {
            'device': self.m_Device,
            'criterion': torch.nn.CrossEntropyLoss(),
            'testDataLoader': testData,
            'bLoad':True,
            'savePath': '/home/yoon/paper/source/model/plot/2021-11-30_2019_rgbEntire.preEfficientNetB5.',
            'loadPath': '/home/yoon/paper/source/model/2021-11-30_2019_rgbEntire.preEfficientNetB5.pt'
        }

        model, bestIdx, bestAcc, trainLoss, trainAcc, validLoss, validAcc = self.m_cPretrainedModel.TrainModel(model, paramsTrain, _numEpochs=self.m_Epochs)

        self.m_cPretrainedModel.PlotGraph(
            _bestIdx = bestIdx,
            _bestAcc = bestAcc,
            _trainLoss = trainLoss,
            _trainAcc = trainAcc,
            _validLoss = validLoss,
            _validAcc = validAcc,
            _savePath = '/home/yoon/paper/source/model/plot/2021-11-30_2019_rgbEntire.preEfficientNetB5.pt.'
        )

        labelList, predictedList = self.m_cPretrainedModel.Test(model, paramsTest, _numImages=10)
        self.m_cPretrainedModel.ConfusionMatricSklearn(self.m_NumClasses, labelList, predictedList)

if __name__ == '__main__':
    mainClass = PretrainedEfficientNetMain()
    mainClass.main()

