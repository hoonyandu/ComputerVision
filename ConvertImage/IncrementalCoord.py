import cv2
# import binascii
import numpy as np
import os
import datetime

class IncrementalCoord:
    def __init__(self):
        pass

    def GetEntire(self, _openPath):
        with open(_openPath, 'rb') as f:
            return f.read()

    def GetCoordData(self, _text):
        # text = binascii.hexlify(_text)
        text = _text.hex()
        coordData = list()
        coordinates = np.zeros((256, 256))

        for i in range(0, len(text), 4):
            # bytesLeft = text[i:i+2].decode('utf-8')
            # bytesRight = text[i+2:i+4].decode('utf-8')
            bytesLeft = text[i:i+2]
            bytesRight = text[i+2:i+4]
            if bytesRight == '':
                coordData.append([int(bytesLeft, 16), 0])
            else:
                coordData.append([int(bytesLeft, 16), int(bytesRight, 16)])

        for coord in coordData:
            x = coord[0]
            y = coord[1]

            coordinates[x][y] += 1

        return coordinates

    def GetReverseCoordData(self, _text):
        text = binascii.hexlify(_text)
        coordData = list()
        coordinates = np.full((256, 256), 255)

        for i in range(0, len(text), 4):
            bytesLeft = text[i:i+2].decode('utf-8')
            bytesRight = text[i+2:i+4].decode('utf-8')
            if bytesRight == '':
                coordData.append([int(bytesLeft, 16), 0])
            else:
                coordData.append([int(bytesLeft, 16), int(bytesRight, 16)])

        for coord in coordData:
            x = coord[0]
            y = coord[1]

            coordinates[x][y] -= 1

        return coordinates

    def NormalizedCoord(self, _coordinates):
        maxCoord = np.max(_coordinates)
        normalized = np.zeros((256, 256))

        for row in range(0, _coordinates.shape[0]):
            for col in range(0, _coordinates.shape[1]):
                if _coordinates[row][col] == 0:
                    continue
                normalized[row][col] = 255 - np.round(_coordinates[row][col] / maxCoord * 255)

        return normalized

    def main(self, _openPath, _savePath):

        nowDate = datetime.datetime.now().strftime('%Y-%m-%d')

        for _, _, files in os.walk(_openPath):

            for f in files:
                openFile = _openPath + '/' + f
                saveFile = _savePath + '/' + nowDate + '_' + f + '.jpg'

                text = self.GetEntire(openFile)
                coordinates = self.GetCoordData(text)
                normalized = self.NormalizedCoord(coordinates)

                # cv2.imwrite(saveFile, coordinates)
                cv2.imwrite(saveFile, normalized)


if __name__ == '__main__':
    
    openPath = '/home/malware/Binary_PE/kisa/2019/Benign'
    savePath = '/home/yoon/paper/image/coord/normalized/benign'

    mainClass = IncrementalCoord()
    mainClass.main(_openPath = openPath, _savePath = savePath)