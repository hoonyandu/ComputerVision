import sys
import os
import datetime
import binascii

import TutorialTest
import Pe2Img

class ConvertImage():
    def __init__(self, _openPath, _savePath):
        self.m_OpenPath = _openPath
        self.m_SavePath = _savePath

        self.m_cTutorialTest = TutorialTest.TutorialTest()
        self.m_cPe2Img = Pe2Img.Pe2Img()

    def SaveImage(self, _peSections, _color, _f, _result):

        peSections = _peSections
        color = _color
        peData = list()

        nowDate = datetime.datetime.now().strftime('%Y-%m-%d')

        openFile = self.m_OpenPath + '/' + _f
        saveFile = self.m_SavePath + _result + '/gray/1st/' + nowDate + '_' + color + '_' + peSections + '_' + _f + '.png'
        print(saveFile)

        try:
            # case of not existed pe sections.
            if peSections == 'Entire':
                text = self.m_cPe2Img.GetEntire(_openPath=openFile)

            elif peSections == 'Text':
                text = self.m_cPe2Img.GetTextSection(_openPath=openFile)

            elif peSections == 'TextData':
                text = self.m_cPe2Img.GetTextDataSection(_openPath=openFile)

            text = binascii.hexlify(text)
            bytesData = [text[i:i+2].decode('utf-8') for i in range(0, len(text), 2)]
            peData = [int(bytesData[i], 16) for i in range(len(bytesData))]
            peData = [peData[i:i+16] for i in range(0, len(peData), 16)]

            if color == 'rgb':
                self.m_cTutorialTest.ConvertRGB(peData, saveFile)

            elif color == 'gray':
                self.m_cTutorialTest.ConvertAndSave(peData, saveFile)

        except Exception as e:
            print(openFile)
            print('Error: ', e)

    def main(self, _argv):

        if (len(_argv) != 3):
            print("Enter \n1. a Pe's Section (Entire, Text, TextData) \n2. a color scale(rgb, gray)")
            sys.exit()

        peSections = _argv[1]
        color = _argv[2]
        peData = list()

        nowDate = datetime.datetime.now().strftime('%Y-%m-%d')
        
        for _, _, files in os.walk(self.m_OpenPath):

            for f in files:
                openFile = self.m_OpenPath + '/' + f
                saveFile = self.m_SavePath + '/' + nowDate + '_' + color + '_' + peSections + '_' + f + '.png'

                try:
                    # case of not existed pe sections.
                    if peSections == 'Entire':
                        text = self.m_cPe2Img.GetEntire(_openPath=openFile)

                    elif peSections == 'Text':
                        text = self.m_cPe2Img.GetTextSection(_openPath=openFile)

                    elif peSections == 'TextData':
                        text = self.m_cPe2Img.GetTextDataSection(_openPath=openFile)

                    text = binascii.hexlify(text)
                    bytesData = [text[i:i+2].decode('utf-8') for i in range(0, len(text), 2)]
                    peData = [int(bytesData[i], 16) for i in range(len(bytesData))]
                    peData = [peData[i:i+16] for i in range(0, len(peData), 16)]

                    if color == 'rgb':
                        self.m_cTutorialTest.ConvertRGB(peData, saveFile)

                    elif color == 'gray':
                        self.m_cTutorialTest.ConvertAndSave(peData, saveFile)

                except Exception as e:
                    print(openFile)
                    print('Error: ', e)



if __name__ == '__main__':
    openPath = '/home/yoon/whitelist/whitelist_binary/EXE'
    savePath = '/home/yoon/paper/image/whitelist'

    mainClass = ConvertImage(
        _openPath = openPath,
        _savePath = savePath
    )
    mainClass.main(sys.argv)




