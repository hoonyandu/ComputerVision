import cv2
import os
import datetime
import numpy as np

class ImageMerge:
    def __init__(self, _originalPath, _gradCamPath, _savePath, _epochStr):
        self.m_OriginalPath = _originalPath
        self.m_GradCamPath = _gradCamPath
        self.m_SavePath = _savePath
        self.m_EpochStr = _epochStr

    def NameMatching(self, _originImg, _gradCamPath):
        original = (_originImg.split('_')[-1]).split('.')[0]

        for _, _, files in os.walk(_gradCamPath):
            for f in files:
                if (original in f) and (self.m_EpochStr in f):
                    return f

        return None

    def LoadMerge(self, _origin, _gradCam, _savePath):

        nowDate = datetime.datetime.now().strftime('%Y-%m-%d')

        originImg = cv2.imread(_origin)
        gradCamImg = cv2.imread(_gradCam)
        # saveName = _savePath + '/' + nowDate + '_' + _gradCam.split('_')[-2] + self.m_EpochStr + '.png'
        saveName = _savePath + '/' + nowDate + '_' + _gradCam.split('2021-06-21_')[-1]

        gradCamImg = cv2.resize(gradCamImg, (originImg.shape[1], originImg.shape[0]))
        gradCamImg = np.uint8(255 * gradCamImg)
        gradCamImg = cv2.applyColorMap(gradCamImg, cv2.COLORMAP_JET)

        saveImg = gradCamImg * 0.4 + originImg

        cv2.imwrite(saveName, saveImg)
        print('save path: ', saveName)


    def main(self):

        classification = ['benign', 'malware']

        for cl in classification:
            originalPath = self.m_OriginalPath + '/' + cl
            gradCamPath = self.m_GradCamPath + '/' + cl
            savePath = self.m_SavePath + '/' + cl

            for _, _, files in os.walk(originalPath):
                
                for origin in files:
                    print('origin: ', origin)
                    gradCam = self.NameMatching(origin, gradCamPath)

                    if gradCam:
                        self.LoadMerge(originalPath + '/' + origin, gradCamPath + '/'+ gradCam, savePath)


if __name__ == '__main__':

    originalPath = '/home/yoon/paper/image/whitelist/test'
    gradCamPath = '/home/yoon/paper/image/whitelist/gradCam'
    savePath = '/home/yoon/paper/image/whitelist/merge'
    epochStr = ''


    mainClass = ImageMerge(
        _originalPath = originalPath,
        _gradCamPath = gradCamPath,
        _savePath = savePath,
        _epochStr = epochStr
    )

    mainClass.main()
        

