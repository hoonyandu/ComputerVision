import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder

class PyTorchData():

    def __init__(self, _dataType, _dataPath, _inputDim, _batchSize):

        if _dataType == "data":
            self.m_DataDim = _inputDim
        elif _dataType == "image":
            self.m_ImageDim = _inputDim
        self.m_DataPath = _dataPath
        self.m_BatchSize = _batchSize

    def ImageTrain(self):

        # normalize the color range to [0,1] : transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        transDatagen = transforms.Compose(
            [
                transforms.Resize((self.m_ImageDim, self.m_ImageDim)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        trainPath = self.m_DataPath + '/training'
        trainFolder = torchvision.datasets.ImageFolder(
                root = trainPath,
                transform = transDatagen
            )

        trainLoader = DataLoader(
                trainFolder,
                batch_size = self.m_BatchSize,
                shuffle = True,
            )

        print("Train Class [", trainLoader.dataset.class_to_idx, "]")
        print("Train Numbers [", len(trainLoader.dataset.imgs), "]")
        print("Train Batch Size [", trainLoader.batch_size, "]")

        trainLoader.requires_grad = True

        return trainLoader

    def ImageValidation(self):
        transDatagen = transforms.Compose(
            [
                transforms.Resize((self.m_ImageDim, self.m_ImageDim)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        validationPath = self.m_DataPath + '/validation'
        validationSet = torchvision.datasets.ImageFolder(
                root = validationPath,
                transform = transDatagen
            )

        validationLoader = DataLoader(
                validationSet,
                batch_size = self.m_BatchSize,
                shuffle = True 
            )

        print("Validation Class [", validationLoader.dataset.class_to_idx, "]")
        print("Validation Numbers [", len(validationLoader.dataset.imgs),"]")
        print("Validation Batch Size [", validationLoader.batch_size,"]")

        validationLoader.requires_grad = True

        return validationLoader

        
    def ImageTest(self):
        transDatagen = transforms.Compose(
            [
                transforms.Resize((self.m_ImageDim, self.m_ImageDim)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]
        )

        testDirectory = self.m_DataPath + '/test'
        testSet = torchvision.datasets.ImageFolder(
                root = testDirectory,
                transform = transDatagen
            )

        testLoader = DataLoader(
                testSet,
                batch_size = self.m_BatchSize,
                shuffle = True 
            )

        print("Test Class [", testLoader.dataset.class_to_idx, "]")
        print("Test Numbers [", len(testLoader.dataset.imgs), "]")
        print("Test Batch Size [", testLoader.batch_size,"]")

        testLoader.requires_grad = True

        return testLoader, testSet.imgs

    def DataSet(self, _testSet):
        transformTest = transforms.Compose([
            transforms.ToTensor()
        ])
        testSet = ImageFolder(root=_testSet, transform=transformTest)

        testSet.requires_grad = True

        return testSet
