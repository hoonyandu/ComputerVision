
import TFDataLoader
import ViTModel
import util

import tensorflow as tf
import os

class ViTMain:
    def __init__(self, _batchSize, _imgHeight, _imgWidth):
        self.m_cDataLoader = TFDataLoader.TFDataLoader(_batchSize, _imgHeight, _imgWidth)

    def CreateModel(self, _learningRate, _params):
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate, weight_decay=weightDecay)
        optimizer = tf.keras.optimizers.Adam(learning_rate=_learningRate)

        model = ViTModel.ImageTransformer(_params = _params)
        model.compile(
            optimizer = optimizer,
            loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
            # metrics=[
            #     # tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            #     tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            #     tf.keras.metrics.AUC( name="AUC"),
            # ],
        )

        return model

    def main(self):

        dirPath = '/Users/hoony/Downloads/normalized-set'
        seed = 0
        savePath = '/Users/hoony/source/tensorflow2/model/220201-ViT'
        saveImg = '/Users/hoony/source/tensorflow2/image/220201/'

        trainDataset = self.m_cDataLoader.ImageTrain(dirPath, seed)
        validationDataset = self.m_cDataLoader.ImageValidation(dirPath, seed)
        testDataset = self.m_cDataLoader.ImageTest(dirPath, seed)

        params = {
            # "imgSize": 32,
            "imgSize": 256,
            # "patchSize": 4,
            "patchSize": 32,
            "dim": 64,
            # "dim": 256,
            "batchSize": 256,
            "depth": 2,
            "heads": 8,
            "mlpDim": 64,
            "nClasses": 2,
        }

        numEpochs = 10
        learningRate = 0.001
        weightDecay = 0.0001
        
        # # optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate, weight_decay=weightDecay)
        # optimizer = tf.keras.optimizers.Adam(learning_rate=learningRate)

        # modelViT = ViTModel.ImageTransformer(_params = params)
        # modelViT.compile(
        #     optimizer = optimizer,
        #     loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        #     metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]
        # )

        # print(modelViT.summary())


        classNames = trainDataset.class_names
        for images, labels in trainDataset.take(1):
            for i in range(3):
                saveImages = saveImg + '/' + str(i)
                # saveImages = os.path.join(saveImg, str(i))
                # os.makedirs(saveImages, exist_ok=True)

                util.ShowImage().ShowPatches(saveImages, images[i], classNames[labels[i]], params["imgSize"], params["patchSize"])

        modelViT = self.CreateModel(learningRate, params)
        # os.makedirs(savePath + "/tmp", exist_ok=True)
        checkpointFilepath = savePath + "/checkpoint.ckpt"
        checkpointCallback = tf.keras.callbacks.ModelCheckpoint(
            checkpointFilepath,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
        )
        # modelViT.fit(trainDataset, batch_size=params["batchSize"], epochs=numEpochs, validation_data=(validationDataset), shuffle=True)
        # history = modelViT.fit(trainDataset, batch_size=params["batchSize"], epochs=numEpochs, validation_data=(validationDataset), shuffle=True)
        history = modelViT.fit(trainDataset, batch_size=params["batchSize"], epochs=numEpochs, validation_data=(validationDataset), shuffle=True, callbacks=[checkpointCallback])

        print('==============Training Finished===============')

        modelViT.save(savePath)

        # loadModel = tf.keras.models.load_model(saveP)
        modelViT.load_weights(checkpointFilepath)
 
        accuracy = 0
        loss, accuracy = modelViT.evaluate(testDataset)
        print('Test Accuracy :', accuracy)

        util.ShowImage().ShowAccuracy(history, saveImg)
        util.ShowImage().ShowLoss(history, saveImg)

if __name__ == "__main__":
    batchSize = 32
    imgHeight = 256
    imgWidth = 256
    # imgHeight = 32
    # imgWidth = 32

    mainClass = ViTMain(batchSize, imgHeight, imgWidth)
    mainClass.main()
