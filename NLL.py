'''
To return NLL for certain parameters of a certain PDF and dataset
'''

from Data import Data
from Function import Function
import math

class NLL(Function):

    def __init__(self, pdf, dataFileName):
        self.pdf = pdf
        self.dataFileName = dataFileName
        data = Data(self.dataFileName)
        self.data = data.data
        self.numData = len(self.data[0])


    def evalNLL(self, params):
        runningNLL = 0.
        for i in range(self.numData):
            L = -math.log(self.pdf(self.data[0][i], params))
            runningNLL +=   L

        return runningNLL
