import torch

import PyTorchData
import GradCamModel
import MakeTestCsv

from torchsummary import summary

from PIL import Image

class GradCamMain():
    def __init__(self):
        self.m_Device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("device type [", self.m_Device, "]")

        self.m_ClassNum = 2
        self.m_BatchSize = 1
        self.m_InputDim = 200
        self.m_LearningRate = 0.0004
        self.m_Epochs = 10

        self.m_cGradCamModel = GradCamModel.CNN(
            _classNum = self.m_ClassNum
        )

        self.m_cPytorchData = PyTorchData.PyTorchData(
            _dataType = "image",
            _dataPath = '/home/yoon/paper/image/whitelist',
            _inputDim = self.m_InputDim,
            _batchSize = self.m_BatchSize
        )

        self.m_cMakeModel = GradCamModel.MakeModel()
        self.m_cMakeCsv = MakeTestCsv.MakeTestCsv(
            _savePath = '/home/yoon/paper/data/2021-06-23_2019_rgbEntire.7.2.1.GradCam_whitelist.csv'
        )

    def main(self):
        print("\n[ Model ]\n")
        model = self.m_cGradCamModel.to(self.m_Device)
        print(model)
        summary(model, (3, self.m_InputDim, self.m_InputDim))

        print("\n[ Data ]\n")
        trainData = self.m_cPytorchData.ImageTrain()
        valData = self.m_cPytorchData.ImageValidation()
        testData, testSetList = self.m_cPytorchData.ImageTest()

        print("\n[ Training ] \n")
        self.m_cMakeModel.GradCam(
            _device = self.m_Device,
            _model = model,
            _epoch = self.m_Epochs,
            _lr = self.m_LearningRate,
            _trainData = trainData,
            _valData = valData,
            _testData = testData,
            _testSetList = testSetList,
            _imgPath = '/home/yoon/paper/image/rgbEntire/2019/gradCam/',
            _bSave = True,
            _savePath = '/home/yoon/paper/source/model/2021-06-07_2019_rgbEntire.7.2.1.GradCam.sav'
        )

        print("\n[ Testing ] \n")
        """ testing """
        labels, predicted = self.m_cMakeModel.TestingAccuracy(
            _device = self.m_Device,
            _model = model,
            _lr = self.m_LearningRate,
            _testData = testData,
            _bLoad = True,
            _loadPath = '/home/yoon/paper/source/model/2021-06-07_2019_rgbEntire.7.2.1.GradCam.sav'
        )

        self.m_cMakeCsv.MakeCsv(
            _labels = labels,
            _predicted = predicted,
            _testSetList = testSetList
        )

        """ gradCam """
        self.m_cMakeModel.GradCamTest(
            _device = self.m_Device,
            _model = model,
            _lr = self.m_LearningRate,
            _testData = testData,
            _testSetList = testSetList,
            _imgPath = '/home/yoon/paper/image/whitelist/gradCam/',
            _bLoad = True,
            _loadPath = '/home/yoon/paper/source/model/2021-06-07_2019_rgbEntire.7.2.1.GradCam.sav'
        )

if __name__ == '__main__':
    mainClass = GradCamMain()
    mainClass.main()