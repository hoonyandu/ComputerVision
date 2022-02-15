import torch
from torch import nn, einsum
from einops import rearrange
import numpy as np

from torchsummary import summary

# https://visionhong.tistory.com/31

class CyclicShift(nn.Module):
    """
    Cyclic Shift

    """
    def __init__(self, _displacement):
        super().__init__()
        self.m_Displacement = _displacement

    def forward(self, _x):
        """
        각 차원별로 3씩 밀어버림
        self.m_Displacement = windowSize // 2
        """
        # _x.shape (b, 56, 56, 96)
        return torch.roll(_x, shifts=(self.m_Displacement, self.m_Displacement), dims=(1, 2))

class Residual(nn.Module):
    """
    Skip Connection

    """
    def __init__(self, _fn):
        super().__init__()
        self.m_Fn = _fn

    def forward(self, _x, **kwargs):
        return self.m_Fn(_x, **kwargs) + _x

class PreNorm(nn.Module):
    """
    Layer Normalization

    """
    def __init__(self, _dim, _fn):
        super().__init__()
        self.m_Norm = nn.LayerNorm(_dim)
        self.m_Fn = _fn

    def forward(self, _x, **kwargs):
        return self.m_Fn(self.m_Norm(_x), **kwargs)

class FeedForward(nn.Module):
    """
    Encoder's MLP

    """
    def __init__(self, _dim, _hiddenDim):
        super().__init__()
        self.m_Net = nn.Sequential(
            nn.Linear(_dim, _hiddenDim),
            nn.GELU(),
            nn.Linear(_hiddenDim, _dim),
        )
        
    def forward(self, _x):
        return self.m_Net(_x)

class GetMask:
    """
    Cyclic Shift 뒤에 수행할 마스킹 작업을 하는 함수

    """
    def __init__(self):
        pass

    def CreateMask(self, _windowSize, _displacement, _upperLower, _leftRight):
        mask = torch.zeros(_windowSize ** 2, _windowSize ** 2)

        if _upperLower:
            masking = _displacement * _windowSize
            mask[-masking:, :-masking] = float('-inf')
            mask[:-masking, -masking:] = float('-inf')

        if _leftRight:
            mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=_windowSize, h2=_windowSize)
            mask[:, -_displacement:, :, :-_displacement] = float('-inf')
            mask[:, :-_displacement, :, -_displacement:] = float('-inf')
            mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

        return mask

    def GetRelativeDistance(self, _windowSize):
        indices = torch.tensor(np.array([[x, y] for x in range(_windowSize) for y in range(_windowSize)]))
        distances = indices[None, :, :] - indices[:, None, :]
        return distances

class WindowAttention(nn.Module):
    """
    W-MSA & SW-MSA
    
    :
    Transformer의 Multi head Self-Attention을 수행하는 class로 
    block의 두번째 encoder인 SW-MSA에서만 self.shifted=True가 되어서 cyclic shift + mask를 진행한다.
    
    RelativePositionEmbedding을 query와 key값의 dot product를 scale로 나눈값에 더해준다.

    """
    def __init__(self, _dim, _heads, _headDim, _shifted, _windowSize, _relativePosEmbedding):
        super().__init__()
        innerDim = _headDim * _heads

        self.m_Heads = _heads
        self.m_Scale = _headDim ** -0.5
        self.m_WindowSize = _windowSize
        self.m_RelativePosEmbedding = _relativePosEmbedding
        self.m_Shifted = _shifted

        if self.m_Shifted:
            displacement = _windowSize // 2 # 7//2 = 3
            self.m_CyclicShift = CyclicShift(-displacement)
            self.m_CyclicBackShift = CyclicShift(displacement)
            self.m_UpperLowerMask = nn.Parameter(GetMask().CreateMask(_windowSize=_windowSize, _displacement=displacement,
                                                                _upperLower=True, _leftRight=False), requires_grad=False)
            self.m_LeftRightMask = nn.Parameter(GetMask().CreateMask(_windowSize=_windowSize, _displacement=displacement,
                                                                _upperLower=False, _leftRight=True), requires_grad=False)
        self.m_ToQKV = nn.Linear(_dim, innerDim * 3, bias=False)
        
        if self.m_RelativePosEmbedding:
            # self.m_RelativeIndices -> index(0~12 사이의 수를 가짐) / + _windowSize - 1 은 음수를 없애기 위해
            self.m_RelativeIndices = GetMask().GetRelativeDistance(_windowSize) + _windowSize - 1
            self.m_PosEmbedding = nn.Parameter(torch.randn(2 * _windowSize - 1, 2 * _windowSize - 1))
        else:
            self.m_PosEmbedding = nn.Parameter(torch.randn(_windowSize ** 2, _windowSize ** 2))

        self.m_ToOut = nn.Linear(innerDim, _dim)

    def forward(self, _x):
        if self.m_Shifted:
            _x = self.m_CyclicShift(_x)

        b, nH, nW, _, h = *_x.shape, self.m_Heads

        qkv = self.m_ToQKV(_x).chunk(3, dim=-1) # (b, 56, 56, 288) -> (b, 56, 56, 96)
        nWH = nH // self.m_WindowSize # 8
        nWW = nW // self.m_WindowSize # 8

        q, k, v = map(
            lambda t: rearrange(t, 'b (nWH wH) (nWW wW) (h d) -> b h (nWH nWW) (wH wW) d',
                                h=h, wH=self.m_WindowSize, wW=self.m_WindowSize), qkv)

        # (b, 3, 64, 49, 32), (b, 3, 64, 49, 32) -> (b, 3, 64, 49, 49)
        # query와 key사이의 연관성 (dot product) * scale(1/root(_headDim))
        dots = einsum('b h w i d, b h w j d -> b h w i j', q, k) * self.m_Scale

        if self.m_RelativePosEmbedding:
            dots = dots + self.m_PosEmbedding[self.m_RelativeIndices[:, :, 0].type(torch.long),
                                                self.m_RelativeIndices[:, :, 1].type(torch.long)] # (49, 49)

        else:
            dots = dots + self.m_PosEmbedding
        
        if self.m_Shifted: # masking
            dots[:, :, -nWW] = dots[:, :, -nWW] + self.m_UpperLowerMask # 아래쪽 가로모양 윈도우
            dots[:, :, nWW - 1::nWW] = dots[:, :, nWW - 1::nWW] + self.m_LeftRightMask # 오른쪽 세로모양 마스킹
        
        attn = dots.softmax(dim=-1)
        out = einsum('b h w i j, b h w j d -> b h w i d', attn, v)
        out = rearrange(out, 'b h (nWH nWW) (wH wW) d -> b (nWH wH) (nWW wW) (h d)',
                                h=h, wH=self.m_WindowSize, wW=self.m_WindowSize, nWH=nWH, nWW=nWW)
        out = self.m_ToOut(out)

        if self.m_Shifted:
            out = self.m_CyclicBackShift(out) # shift한 값을 원래 위치로
        
        return out
        
class SwinBlock(nn.Module):
    """
    Swin Transformer를 구성하는 한개의 Encoder에 대한 구현이다.
    attention과 mlp 모두 Layer Normalization과 Skip Connection 을 먼저한다.
    (attention과 mlp가 뒤에 있다고 먼저 수행하는게 아님!) (위에 Layer Normalization과 Residual 클래스 참고)
    """
    def __init__(self, _dim, _heads, _headDim, _mlpDim, _shifted, _windowSize, _relativePosEmbedding):
        super().__init__()
        self.m_AttentionBlock = Residual(PreNorm(_dim, WindowAttention(_dim=_dim,
                                                                        _heads=_heads,
                                                                        _headDim=_headDim,
                                                                        _shifted=_shifted,
                                                                        _windowSize=_windowSize,
                                                                        _relativePosEmbedding=_relativePosEmbedding)))
        self.m_MLPBlock = Residual(PreNorm(_dim, FeedForward(_dim=_dim, _hiddenDim=_mlpDim)))

    def forward(self, _x):
        x = self.m_AttentionBlock(_x)
        x = self.m_MLPBlock(x)

        return x

class PatchMerging(nn.Module):
    """
    Patch Partition or Patch Merging & Linear Embedding

    :
    _downscalingFactor는 [4, 2, 2, 2]로 이루어져 있기 때문에 stage1에서는 이미지가 패치로
    partition되고 그 이후 stage는 자동으로 patch merging역할을 한다.
    Linear embedding에서 C는 각 stage에서 [96, 192, 384, 768]를 사용한다.
    """
    def __init__(self, _inChannels, _outChannels, _downscalingFactor):
        super().__init__()
        self.m_DownscalingFactor = _downscalingFactor
        self.m_PatchMerge = nn.Unfold(kernel_size=_downscalingFactor, stride=_downscalingFactor, padding=0)
        self.m_Linear = nn.Linear(_inChannels * _downscalingFactor ** 2, _outChannels)

    def forward(self, _x):
        b, c, h, w = _x.shape
        newH, newW = h // self.m_DownscalingFactor, w // self.m_DownscalingFactor # num patches (56 x 56)
        # self.m_PatchMerge(_x) : (b, 48, 3136)
        # self.m_PatchMerge(_x).view(b, -1, newH, newW) : (b, 48, 56, 56)
        # self.m_PatchMerge(_x).view(b, -1, newH, newW).permute(0, 2, 3, 1) : (b, 56, 56, 48)
        x = self.m_PatchMerge(_x).view(b, -1, newH, newW).permute(0, 2, 3, 1)
        x = self.m_Linear(x) # (b, 56, 56, 48), (b, 56, 56, 96)
        return x

class StageModule(nn.Module):
    """
    Stage Module

    :
    각 스테이지는 Patch partition(merging)과 Swin Block으로 이루어져 있으며 stage3에서는 Swin Block이 세번 반복된다.
    
    """
    def __init__(self, _inChannels, _hiddenDimension, _layers, _downscalingFactor, _numHeads, _headDim, _windowSize,
                    _relativePosEmbedding):
        super().__init__()
        assert _layers % 2 ==0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        self.m_PatchPartition = PatchMerging(_inChannels=_inChannels, _outChannels=_hiddenDimension,
                                                _downscalingFactor=_downscalingFactor)
        self.m_Layers = nn.ModuleList([])
        for _ in range(_layers // 2):
            self.m_Layers.append(nn.ModuleList([
                SwinBlock(_dim=_hiddenDimension, _heads=_numHeads, _headDim=_headDim, _mlpDim=_hiddenDimension * 4,
                            _shifted=False, _windowSize=_windowSize, _relativePosEmbedding=_relativePosEmbedding),
                SwinBlock(_dim=_hiddenDimension, _heads=_numHeads, _headDim=_headDim, _mlpDim=_hiddenDimension * 4,
                            _shifted=True, _windowSize=_windowSize, _relativePosEmbedding=_relativePosEmbedding),
            ]))

    def forward(self, _x):
        x = self.m_PatchPartition(_x)
        for regularBlock, shiftedBlock in self.m_Layers:
            x = regularBlock(x)
            x = shiftedBlock(x)

        return x.permute(0, 3, 1, 2) # (4, 56, 56, 96) -> (4, 96, 56, 56)

class SwinTransformer(nn.Module):
    """
    Swin Transformer

    :
    4개의 stage를 거친 후에 나오는 7x7 patch를 average pooling (mean)을 하고 최종 classifier에 넣어 예측을 한다.
    """
    def __init__(self, *, _hiddenDim, _layers, _heads, _channels=3, _numClasses=1000, _headDim=32, _windowSize=7,
                    _downscalingFactors=(4, 2, 2, 2), _relativePosEmbedding=True):
        super().__init__()
        self.m_Stage1 = StageModule(_inChannels=_channels, _hiddenDimension=_hiddenDim, _layers=_layers[0],
                                    _downscalingFactor=_downscalingFactors[0], _numHeads=_heads[0], _headDim=_headDim,
                                    _windowSize=_windowSize, _relativePosEmbedding=_relativePosEmbedding)

        # input shape
        self.m_Stage2 = StageModule(_inChannels=_hiddenDim, _hiddenDimension=_hiddenDim * 2, _layers=_layers[1],
                                    _downscalingFactor=_downscalingFactors[1], _numHeads=_heads[1], _headDim=_headDim,
                                    _windowSize=_windowSize, _relativePosEmbedding=_relativePosEmbedding)

        self.m_Stage3 = StageModule(_inChannels=_hiddenDim * 2, _hiddenDimension=_hiddenDim * 4, _layers=_layers[2],
                                    _downscalingFactor=_downscalingFactors[2], _numHeads=_heads[2], _headDim=_headDim,
                                    _windowSize=_windowSize, _relativePosEmbedding=_relativePosEmbedding)

        self.m_Stage4 = StageModule(_inChannels=_hiddenDim * 4, _hiddenDimension=_hiddenDim * 8, _layers=_layers[3],
                                    _downscalingFactor=_downscalingFactors[3], _numHeads=_heads[3], _headDim=_headDim,
                                    _windowSize=_windowSize, _relativePosEmbedding=_relativePosEmbedding)

        self.m_MLPHead = nn.Sequential(
            nn.LayerNorm(_hiddenDim * 8),
            nn.Linear(_hiddenDim * 8, _numClasses)
        )

    def forward(self, _img):
        # image shape(b, 3, 224, 224)
        x = self.m_Stage1(_img) # (b, 96, 56, 56)
        x = self.m_Stage2(x) # (b, 192, 28, 28)
        x = self.m_Stage3(x) # (b, 384, 14, 14)
        x = self.m_Stage4(x) # (b, 768, 7, 7)

        x = x.mean(dim=[2, 3]) # (b, 768)
        return self.m_MLPHead(x) # (b, classes)

if __name__ == '__main__':
    hiddenDim = 96
    layers = (2, 2, 6, 2)
    heads = (3, 6, 12, 24)

    model = SwinTransformer(_hiddenDim=hiddenDim, _layers=layers, _heads=heads)
    x = torch.randn(4, 3, 224, 224)

    outputs = model(x)
    print(outputs.shape)

    summary(model, input_size=(3, 224, 224), device='cpu')