import torch

import PyTorchModel
import PyTorchData

from torchsummary import summary

from PIL import Image

class PyTorchMain():
    def __init__(self):
        
        self.m_Device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device type [", self.m_Device, "]")

        self.m_ClassNum = 2
        self.m_BatchSize = 16
        self.m_InputDim = 224
        self.m_LearningRate = 0.0004
        self.m_Epochs = 10

        self.m_cPytorchModel = PyTorchModel.PillModel(
        _classNum = self.m_ClassNum,
        )

        self.m_cPytorchData = PyTorchData.PyTorchData(
            _dataType = "image",
            _dataPath = INPUT_IMAGE_PATH,
            _inputDim = self.m_InputDim,
            _batchSize = self.m_BatchSize
        )

        self.m_cMakeModel = PyTorchModel.MakeModel()

    def main(self):

        print("\n[ Model ]\n")
        model = self.m_cPytorchModel.to(self.m_Device)
        print(model)
        summary(model, ( 3, self.m_InputDim, self.m_InputDim ))

        print("\n[ Data ]\n")
        trainData = self.m_cPytorchData.ImageTrain()
        valData = self.m_cPytorchData.ImageValidation()
        testData, _ = self.m_cPytorchData.ImageTest()

        print("\n[ Training ]\n")
        self.m_cMakeModel.Training(
           _device = self.m_Device,
           _model = model,
           _epoch = self.m_Epochs,
           _lr = self.m_LearningRate,
           _trainData = trainData,
           _valData = valData,
           _bSave = True,
           _savePath = SAVE_MODEL_PATH
        )

        print("\n[ Testing ]\n")
        labels, predicted = self.m_cMakeModel.TestingAccuracy(
            _device = self.m_Device,
            _model = model,
            _testData = testData,
            _bLoad = True,
            _loadPath = LOAD_MODEL_PATH
        )

        self.m_cMakeModel.ConfusionMatrixSklearn(labels, predicted)


if __name__ == '__main__':
    mainClass = PyTorchMain()
    mainClass.main()
