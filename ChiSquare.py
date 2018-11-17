'''
Class to return ChiSquare relating to a dataset in data
for given parameters
'''

from Data import Data
from Function import Function

class ChiSquare(Function):

    def __init__(self, func, dataFileName):
        self.func = func
        self.dataFileName = dataFileName
        data = Data(self.dataFileName)
        self.data = data.data
        self.errData = data.errData
        self.numData = len(data.data[0])

    def evalChiSquare(self, params):
        runningChi = 0.
        for i in range(self.numData):
            d = (self.data[1][i] - self.func(self.data[0][i], params)) / self.errData[i]
            runningChi += d**2.

        return runningChi
