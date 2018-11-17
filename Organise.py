'''
Where instances of all the other classes are moved around
to do useful things.
~Roughly~ a method should correspond to a part in the Report outline
'''
import numpy as np
from Function import Function, FixParams
from PDFGen import PDFGen
import matplotlib.pylab as pl

class Organise(object):

    def plotPDF(self):
        F = 0.5
        tau1 = 1.0
        tau2 = 2.0

        numEvents = 10000

        #this is known by my me (knowing the PDF)
        #but is essentially arbitrary (non-negative)
        maxPDFVal = 1.

        tLower = 0.
        tUpper = 10.
        thetaLower = 0.
        thetaUpper = 2.*np.pi
        freeParamRanges = [[tLower, tUpper], [thetaLower, thetaUpper]]

        coupledPDF = Function()
        coupledFixed = FixParams(coupledPDF.fPDF, 5, [F, tau1, tau2], [0,3,4])
        coupledFixedGen = PDFGen(coupledFixed.eval, freeParamRanges, maxPDFVal)

        #t in [0]
        #theta in [1]
        t_thetaVals = []
        for i in range(numEvents):
                t_thetaVals.append(coupledFixedGen.nextBox())

        #--now for plotting
        #-----------------------------------------------------------------

        xVals = [item[0] for item in t_thetaVals]
        yVals = [item[1] for item in t_thetaVals]

        pl.scatter(xVals, yVals)
        pl.show()
