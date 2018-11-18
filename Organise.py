'''
Where instances of all the other classes are moved around
to do useful things.
~Roughly~ a method should correspond to a part in the Report outline
'''
import numpy as np
from scipy import optimize
from Function import Function, FixParams, ComposeFunction, AddFunction
from Optimise import Optimise
from NLL import NLL
from PDFGen import PDFGen
from Data import Data
import matplotlib.pylab as pl
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
        self.plotDistributions(t_thetaValsCoupled, "Coupled")
        self.plotDistributions(t_thetaValsP1, "P1")
        self.plotDistributions(t_thetaValsP2, "P2")

    def fitTVals(self):
        filename = "data/datafile-Xdecay.txt"
        data = Data(filename)
        func = Function()
        dataForNNL = [[item] for item in data.tVals]

        #too much data!!
        #---------------
        partialData = []
        for i in range(10000):
            partialData.append(dataForNNL[i])

        #---------------

        tDataNLL = NLL(func.fThetaIndepPDF, partialData, [1], 4)

        #initial guesses
        fInitial = 0.8
        tau1Initial = 0.9
        tau2Initial = 2.5
        initialGuess = [fInitial, tau1Initial, tau2Initial]

        #bounds
        fBound = (0.00001,0.999999)
        tau1Bound = (0.00001,20.)
        tau2Bound = (0.00001,20.)
        bounds = [fBound, tau1Bound, tau2Bound]

        tolerance = 0.00000001
        soln = optimize.minimize(tDataNLL.evalNLL, initialGuess, bounds=bounds, tol=tolerance, options={'disp': False}).x
        fMin, tau1Min, tau2Min = soln

        print("Max. likelyhood minimum: ")
        print("\tF min:..\t\t" + str(fMin))
        print("\tTau1 min:..\t\t" + str(tau1Min))
        print("\tTau2 min:..\t\t" + str(tau2Min))

        #finding errors
        #----------------------------------------------------------------------
        root = Optimise()
        fJump = 0.001
        tau1Jump = 0.01
        tau2Jump = 0.01
        jumps = [fJump, tau1Jump, tau2Jump]

        fAccuracy = 0.0001
        tau1Accuracy = 0.0001
        tau2Accuracy = 0.000001
        accuracys = [fAccuracy, tau1Accuracy, tau2Accuracy]

        minus = -0.5-tDataNLL.evalNLL(list(soln))
        takeHalf = AddFunction(minus)
        funcToRoot = ComposeFunction(takeHalf.evalAdd, tDataNLL.evalNLL)

        print(funcToRoot.evalCompose(soln))
        fRoot = root.root(funcToRoot.evalCompose, list(soln), jumps, accuracys, [1,2])[0]
        fErr = fRoot - fMin
        print("FErr: " + str(fErr))

        fRange = np.linspace(fMin - fErr, fMin + fErr, 50)
        NLLVals = []

        for val in fRange:
            NLLVals.append(tDataNLL.evalNLL([val, tau1Min, tau2Min]))

        pl.plot(fRange, NLLVals)
        pl.show()




    def genMany(self, generator, numEvents):
        t_thetaVals = []

        for i in range(numEvents):
            #shows a little progress indicator
            sys.stdout.write("\r" + str(100*i/numEvents)[0:3] + "%")
            t_thetaVals.append(generator.nextBox())
            sys.stdout.flush()
        print("")

        return t_thetaVals

    def plotDistributions(self, t_thetaVals, title):
        tVals = [item[0] for item in t_thetaVals]
        thetaVals = [item[1] for item in t_thetaVals]

        pl.figure(1)
        pl.scatter(thetaVals, tVals, s=1)
        pl.title( title + " theta vs t occurence scatter")
        pl.xlabel("t")
        pl.ylabel("theta")

        #plotting individually
        tBins = 70
        pl.figure(2)
        pl.hist(tVals, bins=tBins)
        pl.title( title + " t distribution")
        pl.xlabel("t")
        pl.ylabel("frequency")

        thetaBins = 70
        pl.figure(3)
        pl.hist(thetaVals, bins=thetaBins)
        pl.title( title + " theta distribution")
        pl.xlabel("theta")
        pl.ylabel("frequency")

        pl.show()
