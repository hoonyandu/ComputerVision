from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

class Patches(layers.Layer):
    def __init__(self, _patchSize):
        super(Patches, self).__init__()
        self.m_PatchSize = _patchSize

    def call(self, _images):
        batchSize = tf.shape(_images)[0]
        patches = tf.image.extract_patches(
            images = _images,
            sizes = [1, self.m_PatchSize, self.m_PatchSize, 1],
            strides = [1, self.m_PatchSize, self.m_PatchSize, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID",
        )

        patchDims = patches.shape[-1]
        patches = tf.reshape(patches, [batchSize, -1, patchDims])
        return patches

class ShowImage:
    def __init__(self):
        pass

    def ShowPatches(self, _savePath, _images, _labels, _imageSize, _patchSize):
        plt.figure(figsize=(4, 4))
        # plt.imshow(_images.astype("uint8"))
        plt.imshow(_images.numpy().astype("uint8"))
        plt.title(_labels)
        plt.axis("off")
        plt.savefig(_savePath + '_original.jpg')

        resizedImage = tf.image.resize(tf.convert_to_tensor([_images]), size=(_imageSize, _imageSize))
        patches = Patches(_patchSize)(resizedImage)
        print(f"Image size: {_imageSize} X {_imageSize}")
        print(f"Patch size: {_patchSize} X {_patchSize}")
        print(f"Patches per image: {patches.shape[1]}")
        print(f"Elements per patch: {patches.shape[-1]}")

        nSize = int(np.sqrt(patches.shape[1]))
        plt.figure(figsize=(4, 4))
        for i, patch in enumerate(patches[0]):
            ax = plt.subplot(nSize, nSize, i + 1)
            patchImg = tf.reshape(patch, (_patchSize, _patchSize, 3))
            plt.imshow(patchImg.numpy().astype("uint8"))
            plt.axis("off")
        plt.savefig(_savePath + '_patches.jpg')

    def ShowAccuracy(self, _history, _savePath):
        # list all data in history
        print(_history.history.keys())
        
        # summarize history for accuracy
        plt.figure(figsize=(12, 10))
        plt.plot(_history.history['accuracy'])
        plt.plot(_history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.imshow()
        plt.show()
        plt.savefig(_savePath + '/accuracy.jpg')

    def ShowLoss(self, _history, _savePath):
        # summarize history for loss
        plt.figure(figsize=(12, 10))
        plt.plot(_history.history['loss'])
        plt.plot(_history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()
        plt.show()
        plt.savefig(_savePath + '/loss.jpg')
