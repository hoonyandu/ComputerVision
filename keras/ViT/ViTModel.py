from numpy import matmul
import tensorflow as tf
from tensorflow.keras.activations import gelu
# from typing import List, Tuple

# https://dev-woong.tistory.com/38

class MultiHeadAttention(tf.keras.Model):
    """
    Define Multi Head Attention
    
    """
    def __init__(self, _dim, _heads=8):
        super(MultiHeadAttention, self).__init__()
        self.m_Heads = _heads
        self.m_Dim = _dim

        assert _dim // _heads
        self.m_Depth = _dim // _heads
        self.m_Q = tf.keras.layers.Dense(_dim)
        self.m_K = tf.keras.layers.Dense(_dim)
        self.m_V = tf.keras.layers.Dense(_dim)
        self.m_Dense = tf.keras.layers.Dense(_dim)
        
    def SplitHeads(self, _x, _batchSize):
        _x = tf.reshape(_x, (_batchSize, -1, self.m_Heads, self.m_Depth))
        return tf.transpose(_x, perm=[0, 2, 1, 3])

    def ScaledDotProductAttention(self, _q, _k, _v):
        matmulQK = tf.matmul(_q, _k, transpose_b=True)
        dK = tf.cast(tf.shape(_k)[-1], tf.float32)
        scaledAttentionLogits = matmulQK / tf.math.sqrt(dK)

        softmax = tf.nn.softmax(scaledAttentionLogits, axis=-1)
        scaledDotProductAttentionOutput = tf.matmul(softmax, _v)
        return scaledDotProductAttentionOutput, softmax

    def call(self, _inputs):
        output = None
        batchSize = tf.shape(_inputs)[0]
        q = self.m_Q(_inputs)
        k = self.m_K(_inputs)
        v = self.m_V(_inputs)

        q = self.SplitHeads(q, batchSize)
        k = self.SplitHeads(k, batchSize)
        v = self.SplitHeads(v, batchSize)

        attentionWeights, softmax = self.ScaledDotProductAttention(q, k, v)
        sacledAttention = tf.transpose(attentionWeights, perm=[0, 2, 1, 3])
        concatAttention = tf.reshape(sacledAttention, (batchSize, -1, self.m_Dim))
        output = self.m_Dense(concatAttention)

        return output

class ResidualBlock(tf.keras.Model):
    """
    Define Redisual Block

    """
    def __init__(self, _residualFunction):
        super(ResidualBlock, self).__init__()
        self.m_ResidualFunction = _residualFunction

    def call(self, _inputs):
        return self.m_ResidualFunction(_inputs) + _inputs


class NormalizationBlock(tf.keras.Model):
    """
    Define Layer Normalization

    """
    def __init__(self, _normFunction, _epsilon=1e-5):
        super(NormalizationBlock, self).__init__()
        self.m_NormalFunction = _normFunction
        self.m_Normalize = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, _inputs):
        return self.m_NormalFunction(self.m_Normalize(_inputs))


class MLPBlock(tf.keras.Model):
    """
    Define MLP Block
    
    """
    def __init__(self, _outputDim, _hiddenDim):
        super(MLPBlock, self).__init__()
        self.m_OutputDim = tf.keras.layers.Dense(_outputDim)
        self.m_hiddenDim = tf.keras.layers.Dense(_hiddenDim)
        self.m_Dropout1 = tf.keras.layers.Dropout(0.1)
        self.m_Dropout2 = tf.keras.layers.Dropout(0.1)

    def call(self, _inputs):
        output = None
        x = self.m_hiddenDim(_inputs)
        x = gelu(x)
        x = self.m_Dropout1(x)
        x = self.m_OutputDim(x)
        x = gelu(x)
        output = self.m_Dropout2(x)

        return output


class TransformerEncoder(tf.keras.layers.Layer):
    """
    Define Transformer Encoder

    """
    def __init__(self, _dim, _depth, _heads, _mlpDim, _imgSize, _patchSize):
        super(TransformerEncoder, self).__init__()
        objImgSize = _imgSize
        patchSize = _patchSize

        layers_ = list()
        layers_.append(tf.keras.Input(shape=((objImgSize // patchSize) * (objImgSize // patchSize) + 1, _dim)))
        # layers_.append(tf.keras.layers.InputLayer(shape=((objImgSize // patchSize) * (objImgSize // patchSize) + 1, _dim)))
        for i in range(_depth):
            layers_.append(NormalizationBlock(ResidualBlock(MultiHeadAttention(_dim, _heads))))
            layers_.append(NormalizationBlock(ResidualBlock(MLPBlock(_dim, _mlpDim))))

        self.m_Layers = tf.keras.Sequential(layers_)

    def call(self, _inputs):
        return self.m_Layers(_inputs)


class ImageTransformer(tf.keras.Model):
    """
    Make ViT Model
    
    """
    def __init__(self, _params, _channels=3):
        super(ImageTransformer, self).__init__()
        imgSize = _params["imgSize"]
        patchSize = _params["patchSize"]
        
        assert imgSize % patchSize == 0, 'invalid patch size for image size'

        numPatches = (imgSize // patchSize) ** 2
        self.m_PatchSize = patchSize

        self.m_Dim = _params["dim"]
        self.m_BatchSize = _params["batchSize"]
        self.m_Heads = _params["heads"]
        self.m_Depth = _params["depth"]
        self.m_MlpDim = _params["mlpDim"]
        self.m_NClasses = _params["nClasses"]
        self.m_NumPatches = numPatches

        self.m_PositionalEmbedding = self.add_weight(
            "position_embeddings", shape=[numPatches + 1, _params["dim"]],
            initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32
        )
        # self.m_PositionalEmbedding = self.add_weight(
        #     "position_embeddings", shape=[numPatches, _params["dim"]],
        #     initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32
        # )
        self.m_ClassificationToken = self.add_weight(
            "classification_token", shape=[1, 1, _params["dim"]],
            initializer=tf.keras.initializers.RandomNormal(), dtype=tf.float32
        )

        self.m_PatchProjection = tf.keras.layers.Dense(_params["dim"])
        self.m_Normalization2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.m_MLP = MLPBlock(self.m_Dim, self.m_MlpDim)
        self.m_OutputClasses = tf.keras.layers.Dense(self.m_NClasses)
        self.m_Transformer = TransformerEncoder(self.m_Dim, self.m_Depth, self.m_Heads, self.m_MlpDim, imgSize, patchSize)
        self.m_Dropout1 = tf.keras.layers.Dropout(0.5)

    def call(self, _inputs):
        output = None
        batchSize = tf.shape(_inputs)[0]

        """
        Most Important Part
        """

        # Split a image with patch size
        patches = tf.image.extract_patches(
            images = _inputs,
            sizes = [1, self.m_PatchSize, self.m_PatchSize, 1],
            strides = [1, self.m_PatchSize, self.m_PatchSize, 1],
            rates = [1, 1, 1, 1],
            padding = "VALID",
        )

        patchDims = patches.shape[-1]
        patches = tf.reshape(patches, [batchSize, patches.shape[1]*patches.shape[2], patchDims])
        x = self.m_PatchProjection(patches)

        clsPos = tf.broadcast_to(
            self.m_ClassificationToken, [batchSize, 1, self.m_Dim]
        )

        x = tf.concat([clsPos, x], axis=1)
        x = x + self.m_PositionalEmbedding
        x = self.m_Transformer(x)
        x = self.m_Normalization2(x)
        x = x[:, 0, :]
        xKeep = tf.identity(x)
        x = self.m_Dropout1(x)
        output = self.m_OutputClasses(x)

        return output