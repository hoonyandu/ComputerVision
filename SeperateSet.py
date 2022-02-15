import os
import shutil
import sys

class SeperateSet():
    def __init__(self):
        pass

    def CheckFolderSets(self, _newDirPath, _state):
        folderList = ['training', 'validation', 'test']
        for folder in folderList:
            address = os.path.join(_newDirPath, folder)

            if not os.path.exists(address):
                os.makedirs(address)

            if not os.path.exists(os.path.join(address, _state)):
                os.makedirs(os.path.join(address, _state))


    def SeperateFile(self, _dirPath, _newDirPath, _peSection, _color, _state):
        index = 0
        print(_color + '_' + _peSection)
        for path, folder, files in os.walk(_dirPath):

            for f in files:
                # print(files)

                if _color + '_' + _peSection in f:
                    fullFileName = os.path.join(_dirPath, f)
                    print(fullFileName)

                    ''' 7 : 2 : 1 '''

                    if index % 10 < 7:
                        shutil.copyfile(fullFileName, os.path.join(_newDirPath + '/training/' + _state, f))

                    elif index % 10 >= 7 and index % 10 < 9:
                        shutil.copyfile(fullFileName, os.path.join(_newDirPath + '/validation/' + _state, f))

                    else:
                        shutil.copyfile(fullFileName, os.path.join(_newDirPath + '/test/' + _state, f))

                    index = index + 1



class Main():
    def __init__(self):
        self.m_cSeperateSet = SeperateSet()

    def main(self, _argv):
        if (len(_argv) != 3):
            print('\nusage: python3 SeperateSet.py malware/benign peSection color')
            print('malware/benign - files\' state which are malware or benign')
            print('color - a color scale of images (gray, rgb)\n')
            sys.exit()

        state = _argv[1]
        color = _argv[2]

        dirPath = '/home/yoon/paper/image/' + state + '/gray/3rd/'
        newDirPath = '/home/yoon/paper/image/' + color + '/2019/'

        self.m_cSeperateSet.CheckFolderSets(newDirPath, state)
        self.m_cSeperateSet.SeperateFile(dirPath, newDirPath, peSection, color, state)


if __name__ == '__main__':
    mainClass = Main()
    mainClass.main(sys.argv)
