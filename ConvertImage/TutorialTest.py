import sys
from math import log
import numpy as np
import binascii
import cv2

import Pe2Img


class TutorialTest():
    def __init__(self):
        self.m_cPe2Img = Pe2Img.Pe2Img()

    def ConvertAndSave(self, _peData, _savePath):
        peData = list()
        # print('Processing ' + _openPath)
        if len(_peData[0]) != 16: # If not hexadecimal
            assert(False)

        b = int((len(_peData)*16)**(0.5))
        b = 2**(int(log(b)/log(2))+1)
        a = int(len(_peData)*16/b)

        for i in range(len(_peData)):
            for j in range(len(_peData[i])):
                peData.append(_peData[i][j])

        if len(peData) > (a*b):
            peData = peData[:a*b]

        elif len(peData) < (a*b):
            last = a*b - len(peData)
            for i in last:
                peData.append(00)

        image = np.reshape(np.array(peData), (a, b))

        cv2.imwrite(_savePath, image)

    def ConvertRGB(self, _peData, _savePath):
        peData = list()
        pixelLen = int(len(_peData)*16/3) # 3 rgb values per pixel
        width = int(pixelLen**0.5)
        width = 2**(int(log(width)/log(2))+1)
        height = int(pixelLen/width)

        for i in range(len(_peData)):
            for j in range(len(_peData[i])):
                peData.append(_peData[i][j])

        if len(peData) > (height*width*3):
            peData = peData[:height*width*3]
        
        elif len(peData) < (height*width*3):
            last = height*width*3 - len(peData)
            for i in last:
                peData.append(00)

        rgb = np.reshape(np.array(peData), (height, width, 3))
        cv2.imwrite(_savePath, rgb)

    def main(self, _argv):

        if (len(_argv) != 3):
            print("Enter \n1. a Pe's Section (Entire, Text, TextData) \n2. a color scale(rgb, gray)")
            sys.exit()

        peSections = _argv[1]
        color = _argv[2]
    
        peData = list()

        if peSections == 'Entire':
            text = self.m_cPe2Img.GetEntire()

        elif peSections == 'Text':
            text = self.m_cPe2Img.GetTextSection()

        elif peSections == 'TextData':
            text = self.m_cPe2Img.GetTextDataSection()

        elif peSections == 'Resource':
            text = self.m_cPe2Img.GetRsrcSection()

        text = binascii.hexlify(text)
        bytesData = [text[i:i+2].decode('utf-8') for i in range(0, len(text), 2)]
        peData = [int(bytesData[i], 16) for i in range(len(bytesData))]
        peData = [peData[i:i+16] for i in range(0, len(peData), 16)]
        print(peData[:30])

        if color == 'rgb':
            self.ConvertRGB(peData)

        elif color == 'gray':
            self.ConvertAndSave(peData)

if __name__ == '__main__':

    openPath = '/home/yoon/paper/benign/Binary_PE/test2.vir'
    savePath = '/home/yoon/paper/benign/image/2021-03-16_test2.vir_GetRsrc_rgb.png'

    mainClass = TutorialTest(
        _openPath = openPath,
        _savePath = savePath
        )
    mainClass.main(sys.argv)
