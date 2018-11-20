'''
Where instances of all the other classes are moved around
to do useful things.
~Roughly~ a method should correspond to a part in the Report outline
'''
import numpy as np
from scipy import optimize
import sys
import copy
from iminuit import Minuit

from Function import Function, FixParams, ComposeFunction, AddFunction
from Optimise import Optimise
from NLL import NLL
from PDFGen import PDFGen
from Data import Data
from Plot import Plot

import sys
import copy

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

        #random num generator for F = 0.5
        #----------------------------------------------------------------------
        coupledPDF = Function()
        coupledFixed = FixParams(coupledPDF.fPDF, 5, [F, tau1, tau2], [0,3,4])
        coupledFixedGen = PDFGen(coupledFixed.eval, freeParamRanges, maxPDFVal)

        print("Generating coupled random events...")
        #t in [0]
        #theta in [1]
        t_thetaValsCoupled = self.genMany(coupledFixedGen, numEvents)

        #only P1
        #----------------------------------------------------------------------
        F = 1.
        P1Fixed = FixParams(coupledPDF.fPDF, 5, [F, tau1, tau2], [0,3,4])
        P1FixedGen = PDFGen(P1Fixed.eval, freeParamRanges, maxPDFVal)

        print("Generating P1 only random events...")
        t_thetaValsP1 = self.genMany(P1FixedGen, numEvents)

        #only P2
        #----------------------------------------------------------------------
        F = 0.
        P2Fixed = FixParams(coupledPDF.fPDF, 5, [F, tau1, tau2], [0,3,4])
        P2FixedGen = PDFGen(P2Fixed.eval, freeParamRanges, maxPDFVal)

        print("Generating P2 only random events...")
        t_thetaValsP2 = self.genMany(P2FixedGen, numEvents)

        #now plotting
        #----------------------------------------------------------------------
        plotter = Plot()
        plotter.plotDistributions(t_thetaValsCoupled, "Coupled")
        plotter.plotDistributions(t_thetaValsP1, "P1")
        plotter.plotDistributions(t_thetaValsP2, "P2")

    def fitTVals(self):
        filename = "data/datafile-Xdecay.txt"
        data = Data(filename)
        func = Function()
        dataForNLL = [[item] for item in data.tVals]

        #too much data!! actual had 100K
        #---------------
        partialData = []
        for i in range(50000):
            partialData.append(dataForNLL[i])
        #---------------

        #initial guesses
        fInitial = 0.4
        tau1Initial = 1.9
        tau2Initial = 1.2
        initialGuess = (fInitial, tau1Initial, tau2Initial)

        self.fit(func.fThetaIndepPDF, partialData, [1], 4, initialGuess)
        self.simplisticErrors(self.nllEvaluator, self.soln)

    def fitFull(self):
        filename = "data/datafile-Xdecay.txt"
        data = Data(filename)
        func = Function()
        fullData = zip(data.tVals, data.thetaVals)
        fullData = [list(item) for item in fullData]

        #too much data!! actual had 100K
        #---------------
        partialData = []
        for i in range(50000):
            partialData.append(fullData[i])
        #---------------

        #initial guesses
        fInitial = 0.2
        tau1Initial = 1.6
        tau2Initial = 1.4
        initialGuess = (fInitial, tau1Initial, tau2Initial)

        self.fit(func.fPDF, partialData, [1,2], 5, initialGuess)
        
        opt = Optimise()

        fJump = 0.001
        tau1Jump = 0.01
        tau2Jump = 0.01
        jumps = [fJump, tau1Jump, tau2Jump]

        fAccuracy = 0.0001
        tau1Accuracy = 0.0001
        tau2Accuracy = 0.0001
        accuracys = [fAccuracy, tau1Accuracy, tau2Accuracy]

        fBound = (0.00001,0.999999)
        tau1Bound = (0.00001,20.)
        tau2Bound = (0.00001,20.)
        bounds = (fBound, tau1Bound, tau2Bound)

        posErr = opt.error(self.nllEvaluator.evalNLL, self.soln, 0.5, jumps, accuracys, bounds, ("f", "tau1", "tau2"))
        print(posErr)






    def fit(self, pdf, data, dataParamsIndex, numPDFParams, initialGuess):
        nllCalc = NLL(pdf, data, dataParamsIndex, numPDFParams)

        #bounds
        fBound = (0.00001,0.999999)
        tau1Bound = (0.00001,20.)
        tau2Bound = (0.00001,20.)
        bounds = (fBound, tau1Bound, tau2Bound)

        #tolerances
        fTolerance = 0.001
        tau1Tolerance = 0.001
        tau2Tolerance = 0.001
        tolerances = (fTolerance, tau1Tolerance, tau2Tolerance)

        m = Minuit.from_array_func(nllCalc.evalNLL, initialGuess, error=tolerances, limit=bounds,
                                    name = ("f", "tau1", "tau2"), errordef=0.5)

        m.migrad()
        soln = [value for (key,value) in m.values.items()]
        NLLMin = nllCalc.evalNLL(soln)
        fMin, tau1Min, tau2Min = soln

        print("Vals for NLL minimum: ")
        print("------------------------------------------")
        print("F:\t" + str(fMin))
        print("Tau1:\t" + str(tau1Min))
        print("Tau2:\t" + str(tau2Min))
        print("")

        self.nllEvaluator = nllCalc
        self.soln = soln

    def simplisticErrors(self, nllCalc, soln):
        fMin, tau1Min, tau2Min = soln
        NLLMin = nllCalc.evalNLL(soln)
        #finding errors
        #----------------------------------------------------------------------
        root = Optimise()
        fJump = 0.001
        tau1Jump = 0.01
        tau2Jump = 0.01
        jumps = [fJump, tau1Jump, tau2Jump]

        fAccuracy = 0.0001
        tau1Accuracy = 0.0001
        tau2Accuracy = 0.0001
        accuracys = [fAccuracy, tau1Accuracy, tau2Accuracy]

        shiftVal = 0.5 + NLLMin
        '''
        shift = lambda a: a - shiftVal
        rootShift = ComposeFunction(shift, nllCalc.evalNLL)
        '''

        #upperErrors
        fRoot = root.equalTo(nllCalc.evalNLL, shiftVal, list(soln), jumps, accuracys, [1,2])[0]
        tau1Root = root.equalTo(nllCalc.evalNLL, shiftVal, list(soln), jumps, accuracys, [0,2])[1]
        tau2Root = root.equalTo(nllCalc.evalNLL, shiftVal, list(soln), jumps, accuracys, [0,1])[2]

        fPosErr = fRoot - fMin
        tau1PosErr = tau1Root - tau1Min
        tau2PosErr = tau2Root - tau2Min

        #lowerErrors
        jumpsNeg = [-item for item in jumps]

        fRoot = root.equalTo(nllCalc.evalNLL, shiftVal, list(soln), jumpsNeg, accuracys, [1,2])[0]
        tau1Root = root.equalTo(nllCalc.evalNLL, shiftVal, list(soln), jumpsNeg, accuracys, [0,2])[1]
        tau2Root = root.equalTo(nllCalc.evalNLL, shiftVal, list(soln), jumpsNeg, accuracys, [0,1])[2]

        fNegErr = fMin - fRoot
        tau1NegErr = tau1Min - tau1Root
        tau2NegErr = tau2Min - tau2Root

        fMeanErr = 0.5*(fNegErr + fPosErr)
        tau1MeanErr = 0.5*(tau1NegErr + tau1PosErr)
        tau2MeanErr = 0.5*(tau2NegErr + tau2PosErr)

        print("Errors for NLL minimum (simplistic): ")
        print("------------------------------------------")
        print("F   :\t +" + str(fPosErr) + "\t-" + str(fNegErr) + "\t mean:" + str(fMeanErr))
        print("Tau1:\t +" + str(tau1PosErr) + "\t-" + str(tau1NegErr) + "\t mean:" + str(tau1MeanErr))
        print("Tau2:\t+" + str(tau2PosErr) + "\t-" + str(tau2NegErr) + "\t mean:" + str(tau2MeanErr))
        print("")

        #plotting error info
        numXVals = 50
        fRange = np.linspace(fMin - 2*fNegErr, fMin + 2*fPosErr, numXVals)
        tau1Range = np.linspace(tau1Min - 2*tau1NegErr, tau1Min + 2*tau1PosErr, numXVals)
        tau2Range = np.linspace(tau2Min - 2*tau2NegErr, tau2Min + 2*tau2PosErr, numXVals)

        fYvals = [nllCalc.evalNLL([val, tau1Min, tau2Min]) for val in fRange]
        tau1Yvals = [nllCalc.evalNLL([fMin, val, tau2Min]) for val in tau1Range]
        tau2Yvals = [nllCalc.evalNLL([fMin, tau1Min, val]) for val in tau2Range]

        #centering around minimum
        fRangeCen = [val-fMin for val in fRange]
        tau1RangeCen = [val-tau1Min for val in tau1Range]
        tau2RangeCen = [val-tau2Min for val in tau2Range]

        fYvalsCen = [val-NLLMin for val in fYvals]
        tau1YvalsCen = [val-NLLMin for val in tau1Yvals]
        tau2YvalsCen = [val-NLLMin for val in tau2Yvals]

        plotter = Plot()
        plotter.errorInfo(fRangeCen, fYvalsCen, [fNegErr, fPosErr], 'F')
        plotter.errorInfo(tau1RangeCen, tau1YvalsCen, [tau1NegErr, tau1PosErr], r"$\tau_{1}$")
        plotter.errorInfo(tau2RangeCen, tau2YvalsCen, [tau2NegErr, tau2PosErr], r"$\tau_{2}$")

    def genMany(self, generator, numEvents):
        t_thetaVals = []

        for i in range(numEvents):
            #shows a little progress indicator
            sys.stdout.write("\r" + str(100*i/numEvents)[0:3] + "%")
            t_thetaVals.append(generator.nextBox())
            sys.stdout.flush()
        print("")

        return t_thetaVals
