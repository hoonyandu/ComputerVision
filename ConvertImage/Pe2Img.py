import math
import pefile
import cv2
import numpy as np
from PIL import Image
import sys
import binascii


class Pe2Img():
    def __init__(self):
        pass

    def GetEntire(self, _openPath):
        with open(_openPath, 'rb') as f:
            return f.read()

    def GetTextSection(self, _openPath):
        pe = pefile.PE(_openPath)

        for i in range(len(pe.sections)):
            if b'.text' in pe.sections[i].Name:
                return pe.sections[i].get_data()

    def GetDataSection(self, _openPath):
        pe = pefile.PE(_openPath)

        for i in range(len(pe.sections)):
            if b'.data' in pe.sections[i].Name:
                return pe.sections[i].get_data()

    def GetTextDataSection(self, _openPath):
        pe = pefile.PE(_openPath)

        textData = str()
        for i in range(len(pe.sections)):
            if b'.text' in pe.sections[i].Name:
                textData += str(pe.sections[i].get_data())[2:]

            elif b'.data' in pe.sections[i].Name:
                textData += str(pe.sections[i].get_data())[2:]

        textData = bytes(textData, 'utf-8')

        return textData

    def GetRsrcSection(self, _openPath):
        pe = pefile.PE(_openPath)

        for i in range(len(pe.sections)):
            if b'.rsrc' in pe.sections[i].Name:
                return pe.sections[i].get_data()

    def Bytes2Rgb(self, _text):
        prepend = '\x00\r\n\r\n'
        formedString = prepend + str(_text)[2:-1] + '\x00'

        pixelLen = math.ceil(len(formedString) / 3) # 3 rgb values per pixel
        width = math.ceil(math.sqrt(pixelLen))
        width = math.ceil(width / 4) * 4
        height = math.ceil(pixelLen/width)

        while len(formedString) != (width * height * 3): # Padding
            formedString += '\x00'

        offset = 0
        matrix = [[0 for x in range(width)] for y in range(height)]
        for y in range(height-1, -1, -1): # Loop from the bottom to top
            for x in range(0, width): # Loop from left to right
                r = ord(formedString[offset+2])
                g = ord(formedString[offset+1])
                b = ord(formedString[offset])
                matrix[y][x] = (r, g, b)
                offset += 3

        return matrix

    def Matrix2Bmp(self, _matrix, _savePath):
        height = len(_matrix)
        width = len(_matrix[0])
        image = Image.new("RGB", (width, height))
        pixels = image.load()

        for y in range(height):
            for x in range(width):
                pixels[x, y] = _matrix[y][x]

        try:
            image.save(_savePath, "bmp")
        except:
            return False

        return True

    def Bytes2Gray(self, _text, _savePath):
        # Data length in bytes
        dataLen = len(_text)

        # dataVector is a vector of dataLen bytes
        # dataVector = np.frombuffer(_text, dtype=np.uint8)
        dataVector = np.array(_text, dtype=np.uint8)

        # Assume image shape should be close to square
        sqrtLen = int(math.ceil(math.sqrt(dataLen))) # Compute square root and round up

        # Required length in bytes
        newLen = sqrtLen * sqrtLen

        # Number of bytes to pad (need to add zeros to the end of dataVector)
        padLen = newLen - dataLen

        # Pad dataVector with zeros at the end
        # paddedData = np.pad(dataVector, (0, padLen))
        paddedData = np.hstack((dataVector, np.zeros(padLen, np.uint8)))

        # Reshape 1D array into 2D array with sqrtLen * sqrtLen (image is goint to be a Grayscale image)
        image = np.reshape(paddedData, (sqrtLen, sqrtLen))

        try:
            cv2.imwrite(_savePath, image)
        except:
            return False

        return True

    def main(self, _argv, _savePath):
        if (len(_argv) != 3):
            print("Enter \n1. a Pe's Section (Entire, Text, TextData) \n2. a color scale(rgb, gray)")
            sys.exit()

        peSections = _argv[1]
        color = _argv[2]

        if peSections == 'Entire':
            text = self.GetEntire()
        elif peSections == 'Text':
            text = self.GetTextSection()
        elif peSections == 'TextData':
            text = self.GetTextDataSection()

        text = binascii.hexlify(text)
        bytesData = [text[i:i+2].decode('utf-8') for i in range(0, len(text), 2)]
        peData = [int(bytesData[i], 16) for i in range(len(bytesData))]

        if color == 'rgb':
            matrix = self.Bytes2Rgb(peData)

            if self.Matrix2Bmp(matrix):
                print("Saved as RGB bitmap in: {}".format(_savePath))

        elif color == 'gray':
            if self.Bytes2Gray(peData):
                print("Saved as Grayscale image in: {}".format(_savePath))

if __name__ == '__main__':

    openPath = '/home/yoon/paper/malware/Binary_PE/2018_a10439475e5a314216956285c30afeee.vir'
    savePath = '/home/yoon/paper/malware/image/2021-03-15_2018_a10439475e5a314216956285c30afeee.vir_GetEntire_rgb.bmp'

    mainClass = Pe2Img(_openPath=openPath, _savePath=savePath)
    mainClass.main(sys.argv)

