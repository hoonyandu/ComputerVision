import torch
from torchsummary import summary

import PyTorchData
import EfficientNetModel

class EfficientNetMain():
    def __init__(self):

        self.m_Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print("device type [", self.m_Device, "]")

        self.m_NumClasses = 2
        self.m_InputDim = 224
        self.m_BatchSize = 32
        self.m_LearningRate = 0.01
        self.m_Epochs = 10

        self.m_cEfficientNet = EfficientNetModel.EfficientNetBlock()
        
        self.m_cPytorchData = PyTorchData.PyTorchData(
            _dataType = "image",
            _dataPath = '/home/yoon/paper/image/rgbEntire/2019/',
            _inputDim = self.m_InputDim,
            _batchSize = self.m_BatchSize
        )

        self.m_cMakeModel = EfficientNetModel.MakeModel()

    def main(self):
        print("\n[ Model ]\n")
        model = self.m_cEfficientNet.EfficientNetB2(self.m_NumClasses).to(self.m_Device)
        print(model)
        summary(model, (3, self.m_InputDim, self.m_InputDim), device = self.m_Device.type)

        print("\n[ Data ]\n")
        trainData = self.m_cPytorchData.ImageTrain()
        valData = self.m_cPytorchData.ImageValidation()
        testData, _ = self.m_cPytorchData.ImageTest()

        print("\n[ Training ]\n")
        paramsTrain = {
            'device':self.m_Device,
            'numEpochs':self.m_Epochs,
            'trainDataLoader':trainData,
            'validationDataLoader':valData,
            'sanityCheck':False,
            'pathToWeights':'/home/yoon/paper/source/model/2021-11-15_2019_rgbEntire.7.2.1.EfficientNetB2.pt'
        }
        
        model, lossHistory, metricHistory = self.m_cMakeModel.TrainVal(model, paramsTrain, self.m_LearningRate)

        self.m_cMakeModel.PlotGraph(
            _numEpochs = self.m_Epochs,
            _lossHistory = lossHistory,
            _metricHistory = metricHistory,
            _savePath = '/home/yoon/paper/source/model/plot/2021-11-15_2019_rgbEntire.7.2.1.EfficientNetB1.'
        )

        print("\n[ Test ]\n")
        paramsTest = {
            'device':self.m_Device,
            'testDataLoader':testData,
            'sanityCheck':False,
            'bLoad':True,
            'loadPath':'/home/yoon/paper/source/model/2021-11-15_2019_rgbEntire.7.2.1.EfficientNetB2.pt'
        }

        labelList, predList = self.m_cMakeModel.Testing(model, paramsTest)
        self.m_cMakeModel.ConfusionMatricSklearn(self.m_NumClasses, labelList, predList)

    
if __name__ == '__main__':
    mainClass = EfficientNetMain()
    mainClass.main()