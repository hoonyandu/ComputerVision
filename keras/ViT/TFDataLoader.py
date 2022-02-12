import numpy as np
import os
import tensorflow as tf
import tensorflow_datasets as tfds

class TFDataLoader:
    def __init__(self, _batchSize, _imgHeight, _imgWidth):
        self.m_BatchSize = _batchSize
        self.m_ImgHeight = _imgHeight
        self.m_ImgWidth = _imgWidth

    def CheckRunning(self):
        print("tensorflow version: ", tf.__version__)

    def ImageTrain(self, _dirPath, _seed):
        dirPath = os.path.join(_dirPath, "training")
        trainLoader = tf.keras.preprocessing.image_dataset_from_directory(
            dirPath,
            seed = _seed,
            image_size=(self.m_ImgHeight, self.m_ImgWidth),
            batch_size=self.m_BatchSize
        )
        
        print("Train Class ", trainLoader.class_names)
        print("Train Numbers [", len(trainLoader.file_paths), "]")
        print("Train Batch Size [", self.m_BatchSize, "]")

        return trainLoader

    def ImageValidation(self, _dirPath, _seed):
        dirPath = os.path.join(_dirPath, "validation")
        validationLoader = tf.keras.preprocessing.image_dataset_from_directory(
            dirPath,
            seed = _seed,
            image_size=(self.m_ImgHeight, self.m_ImgWidth),
            batch_size=self.m_BatchSize
        )

        print("Validation Class ", validationLoader.class_names)
        print("Validation Numbers [", len(validationLoader.file_paths), "]")
        print("Validation Batch Size [", self.m_BatchSize, "]")

        return validationLoader

    def ImageTest(self, _dirPath, _seed):
        dirPath = os.path.join(_dirPath, "test")
        testLoader = tf.keras.preprocessing.image_dataset_from_directory(
            dirPath,
            seed = _seed,
            image_size=(self.m_ImgHeight, self.m_ImgWidth),
            batch_size=self.m_BatchSize
        )

        print("Test Class ", testLoader.class_names)
        print("Test Numbers [", len(testLoader.file_paths), "]")
        print("Test Batch Size [", self.m_BatchSize, "]")

        return testLoader
    

if __name__ == '__main__':
    batchSize = 32
    imgHeight = 256
    imgWidth = 256

    dirPath = '/Users/hoony/Downloads/normalized-set'
    seed = 0

    mainClass = TFDataLoader(batchSize, imgHeight, imgWidth)
    mainClass.CheckRunning()
    mainClass.ImageTrain(dirPath, seed)
