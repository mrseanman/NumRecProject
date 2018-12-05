from Data import Data
from Function import Function, FixParams
import math
import copy
import numpy as np

'''
An instance of this class is used to return the value of
the NLL (negative log likelyhood) of a particular PDF
attatched to a dataset.

An instance of NLL can only be used to evaluate the NLL
for one PDF with one set of free params and one dataset.

The PDF values attributed to the data are fixed using FixParams
class and evalNLL then takes the remaining free parameters as arguments.

The idea behind this class was to make evalNLL as fast as
possible. Therefore an instance of NLL is a bit memory
heavy (at least compared to other ways of doing the same thing)
'''
class NLL(Function):

    def __init__(self, pdf, data, dataParamsIndex, numPDFParams):
        self.pdf = pdf
        self.numPDFParams = numPDFParams
        self.data = data
        self.numData = len(self.data)
        self.dataParamsIndex = dataParamsIndex
        self.formFixedPDFs()

    #calls all the elements in self.dataFixedPDFs with the
    #remaining free params in params.
    #FIXME
    def evalNLL(self, params):
        vals = np.empty(self.numData)
        for i in range(self.numData):
            vals[i] = (-np.log(self.dataFixedPDFs[i].eval(params)))
        return np.sum(vals)

    #takes data attatched to PDFs and uses it to form a big list of
    #fixed parameter PDFs (one for every data point)
    #These now only need to be called with the remaining free params.
    def formFixedPDFs(self):
        fixedParamPDFs = []
        for i in range(self.numData):
            singleDataFixedPDF = FixParams(self.pdf, self.numPDFParams, self.data[i], self.dataParamsIndex)
            fixedParamPDFs.append(copy.deepcopy(singleDataFixedPDF))
        self.dataFixedPDFs = fixedParamPDFs
