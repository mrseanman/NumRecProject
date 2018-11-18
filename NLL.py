'''
To return NLL for certain parameters of a certain PDF and dataset
'''

from Data import Data
from Function import Function, FixParams
import math
import copy

class NLL(Function):

    def __init__(self, pdf, data, dataParamsIndex, numPDFParams):
        self.pdf = pdf
        self.numPDFParams = numPDFParams
        self.data = data
        self.numData = len(self.data)
        self.dataParamsIndex = dataParamsIndex
        self.formFixedPDFs()

    def evalNLL(self, params):
        runningNLL = 0.
        for i in range(self.numData):
            L = -math.log(self.dataFixedPDFs[i].eval(params))
            runningNLL +=   L
        return runningNLL

    def formFixedPDFs(self):
        fixedParamPDFs = []
        for i in range(self.numData):
            singleDataFixedPDF = FixParams(self.pdf, self.numPDFParams, self.data[i], self.dataParamsIndex)
            fixedParamPDFs.append(copy.deepcopy(singleDataFixedPDF))
        self.dataFixedPDFs = fixedParamPDFs
