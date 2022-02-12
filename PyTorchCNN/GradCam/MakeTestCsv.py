import csv

class MakeTestCsv():
    def __init__(self, _savePath):
        self.m_SavePath = _savePath

    def MakeCsv(self, _labels, _predicted, _testSetList):

        csvFile = open(self.m_SavePath, 'w', encoding='utf-8', newline='')
        writeCsv = csv.writer(csvFile)

        writeCsv.writerow(['filename', 'label', 'prediction'])

        for idx in range(len(_testSetList)):
            filename = _testSetList[idx][0].split('/')[-1]

            writeCsv.writerow([filename, _labels[idx], _predicted[idx]])

        csvFile.close()

