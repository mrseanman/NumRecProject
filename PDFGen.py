import numpy as np

class PDFGen(object):

    #paramRanges to be given as...
    #[[p1Lower, p1upper], [p2lower, p2upper], ....]
    def __init__(self, pdf, paramRanges, maxPDFVal):
        self.pdf = pdf
        self.numParams = len(paramRanges)
        self.paramRanges = paramRanges
        self.maxPDFVal = maxPDFVal

    #returns value of pdf at parameters given
    def evaluate(self, params):
        return self.pdf(params)

    #returns random value in range according to PDF
    #uses box method
    def nextBox(self):
        #boolean for whether we have found a value to return
        foundVal = False
        while not(foundVal):
            evalParams = []
            for i in range(self.numParams):
                evalParams.append(np.random.uniform(self.paramRanges[i][0],self.paramRanges[i][1]))
            y = np.random.uniform(0., self.maxPDFVal)

            if self.pdf(evalParams) <= y:
                foundVal = True

        return evalParams
