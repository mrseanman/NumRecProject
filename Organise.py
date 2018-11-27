import numpy as np
from iminuit import Minuit

from Function import Function, FixParams, ComposeFunction
from Optimise import Optimise
from NLL import NLL
from RanGen import RanGen
from Data import Data
from Plot import Plot

'''
Where numbers and instances of all the other classes are moved around
to do useful things.
~Roughly~ a method should correspond to a part in the Report outline

Look here is you want to change the starting values of parameters or
to change other minimiser parameters.
'''
class Organise(object):

    def plotPDF(self):
        F = 0.5
        tau1 = 1.0
        tau2 = 2.0

        numEvents = 10000

        #this is known by me (knowing the PDF)
        #a lazy choice (1 is quite above the max PDF val)
        maxPDFVal = 1.

        tLower = 0.
        tUpper = 10.
        thetaLower = 0.
        thetaUpper = 2.*np.pi
        freeParamRanges = [[tLower, tUpper], [thetaLower, thetaUpper]]

        #random num generator for F = 0.5
        #----------------------------------------------------------------------
        coupledPDF = Function()
        coupledFixed = FixParams(coupledPDF.fPDF_tRange, 5, [F, tau1, tau2], [0,3,4])
        coupledFixedGen = RanGen(coupledFixed.eval, freeParamRanges, maxPDFVal)

        print("Generating coupled random events...")
        #t in [0]
        #theta in [1]
        t_thetaValsCoupled = coupledFixedGen.manyBox(numEvents)

        #only P1
        #----------------------------------------------------------------------
        F = 1.
        P1Fixed = FixParams(coupledPDF.fPDF, 5, [F, tau1, tau2], [0,3,4])
        P1FixedGen = RanGen(P1Fixed.eval, freeParamRanges, maxPDFVal)

        print("Generating P1 only random events...")
        t_thetaValsP1 = P1FixedGen.manyBox(numEvents)

        #only P2
        #----------------------------------------------------------------------
        F = 0.
        P2Fixed = FixParams(coupledPDF.fPDF, 5, [F, tau1, tau2], [0,3,4])
        P2FixedGen = RanGen(P2Fixed.eval, freeParamRanges, maxPDFVal)

        print("Generating P2 only random events...")
        t_thetaValsP2 = P2FixedGen.manyBox(numEvents)

        #plotting
        #----------------------------------------------------------------------
        plotter = Plot()
        plotter.plotDistributions(t_thetaValsCoupled, "Coupled")
        plotter.plotDistributions(t_thetaValsP1, "P1")
        plotter.plotDistributions(t_thetaValsP2, "P2")

    #finds F, tau1, tau2 values to fit the fThetaIndepPDF to the t data only
    #does some plots
    def fitTVals(self):
        filename = "data/datafile-Xdecay.txt"
        data = Data(filename)
        func = Function()
        dataForNLL = np.array([[item] for item in data.tVals])

        #initial guesses
        fInitial = 0.962
        tau1Initial = 1.925
        tau2Initial = 0.513
        initialGuess = (fInitial, tau1Initial, tau2Initial)

        print("Initial Guesses:")
        print(initialGuess)

        nllCalc = NLL(func.fThetaIndepPDF, dataForNLL, [1], 4)

        self.fit(nllCalc, initialGuess)

        #plotting error contours (takes very long)
        plotter = Plot()
        print("Plots:\n------------------------")
        plotter.errorCtr(self.minuit, self.fitSoln)

        self.simplisticErrors(nllCalc, self.fitSoln)

        #full errors___________________________________________________
        opt = Optimise()

        fJump = 0.01
        tau1Jump = 0.01
        tau2Jump = 0.01
        #posJumps = [fJump, tau1Jump, tau2Jump]
        posJumps = np.array([fJump, tau1Jump, tau2Jump])

        fAccuracy = 0.00001
        tau1Accuracy = 0.00001
        tau2Accuracy = 0.00001
        #accuracys = [fAccuracy, tau1Accuracy, tau2Accuracy]
        accuracys = np.array([fAccuracy, tau1Accuracy, tau2Accuracy])

        fBound = (0.00001,0.999999)
        tau1Bound = (0.00001,20.)
        tau2Bound = (0.00001,20.)
        bounds = (fBound, tau1Bound, tau2Bound)

        print("PosErrs:\n------------------------")
        posErr = opt.error(nllCalc.evalNLL, self.fitSoln, 0.5,
            posJumps, accuracys, bounds, ("f", "tau1", "tau2"))
        fPosErr, tau1PosErr, tau2PosErr = posErr

        print("NegErrs:\n------------------------")
        negJumps = [-item for item in posJumps]
        negErr = opt.error(nllCalc.evalNLL, self.fitSoln, 0.5,
            negJumps, accuracys, bounds, ("f", "tau1", "tau2"))
        fNegErr, tau1NegErr, tau2NegErr = [-val for val in negErr]

        fMeanErr = 0.5*(fNegErr + fPosErr)
        tau1MeanErr = 0.5*(tau1NegErr + tau1PosErr)
        tau2MeanErr = 0.5*(tau2NegErr + tau2PosErr)

        print("")
        print("Errors for NLL minimum (full): ")
        print("------------------------------------------")
        print("F   :\t+" + str(fPosErr) + "\t-" +
            str(fNegErr) + "\t mean:" + str(fMeanErr))
        print("Tau1:\t+" + str(tau1PosErr) + "\t-" +
            str(tau1NegErr) + "\t mean:" + str(tau1MeanErr))
        print("Tau2:\t+" + str(tau2PosErr) + "\t-" +
            str(tau2NegErr) + "\t mean:" + str(tau2MeanErr))
        print("")

    #finds F, tau1, tau2 values to fit the fPDF to the full (t, theta) data
    def fitFull(self):
        filename = "data/datafile-Xdecay.txt"
        data = Data(filename)
        func = Function()
        fullData = zip(data.tVals, data.thetaVals)
        fullData = np.array(fullData)

        #initial guesses
        fInitial = 0.6
        tau1Initial = 2.40
        tau2Initial = 1.71
        initialGuess = (fInitial, tau1Initial, tau2Initial)

        nllCalc = NLL(func.fPDF, fullData, [1,2], 5)

        self.fit(nllCalc, initialGuess)


        self.simplisticErrors(nllCalc, self.fitSoln)

        #plots error contours (takes very long!)
        plotter = Plot()
        print("Plots:\n------------------------")
        plotter.errorCtr(self.minuit, self.fitSoln)

        #Full errors__________________________________________________
        opt = Optimise()

        fJump = 0.1
        tau1Jump = 0.1
        tau2Jump = 0.1
        posJumps = [fJump, tau1Jump, tau2Jump]

        fAccuracy = 0.000001
        tau1Accuracy = 0.000001
        tau2Accuracy = 0.000001
        accuracys = [fAccuracy, tau1Accuracy, tau2Accuracy]

        fBound = (0.00001,0.999999)
        tau1Bound = (0.00001,20.)
        tau2Bound = (0.00001,20.)
        bounds = (fBound, tau1Bound, tau2Bound)

        print("PosErrs:\n------------------------")
        posErr = opt.error(nllCalc.evalNLL, self.fitSoln, 0.5,
            posJumps, accuracys, bounds, ("f", "tau1", "tau2"))
        fPosErr, tau1PosErr, tau2PosErr = posErr

        print("NegErrs:\n------------------------")
        negJumps = [-item for item in posJumps]
        negErr = opt.error(nllCalc.evalNLL, self.fitSoln, 0.5,
            negJumps, accuracys, bounds, ("f", "tau1", "tau2"))
        fNegErr, tau1NegErr, tau2NegErr = [-val for val in negErr]

        fMeanErr = 0.5*(fNegErr + fPosErr)
        tau1MeanErr = 0.5*(tau1NegErr + tau1PosErr)
        tau2MeanErr = 0.5*(tau2NegErr + tau2PosErr)

        print("")
        print("Errors for NLL minimum (full): ")
        print("------------------------------------------")
        print("F   :\t+" + str(fPosErr) + "\t-" + str(fNegErr) +
            "\t mean:" + str(fMeanErr))
        print("Tau1:\t+" + str(tau1PosErr) + "\t-" + str(tau1NegErr) +
            "\t mean:" + str(tau1MeanErr))
        print("Tau2:\t+" + str(tau2PosErr) + "\t-" + str(tau2NegErr) +
            "\t mean:" + str(tau2MeanErr))
        print("")

    #general method for organising a minuit fit of an NLL
    #
    #nllCalc should be an instance of NLL
    def fit(self, nllCalc, initialGuess):

        fBound = (0.00001,0.999999)
        tau1Bound = (0.00001,20.)
        tau2Bound = (0.00001,20.)
        bounds = (fBound, tau1Bound, tau2Bound)

        fTolerance = 0.001
        tau1Tolerance = 0.001
        tau2Tolerance = 0.001
        tolerances = (fTolerance, tau1Tolerance, tau2Tolerance)

        m = Minuit.from_array_func(nllCalc.evalNLL, initialGuess,
            error=tolerances, limit=bounds, name = ("f", "tau1", "tau2"),
            errordef=0.5, print_level=0)

        m.migrad()

        print("NLL val at minimum:")
        print(m.fval)
        print("")

        soln = np.array([value for (key,value) in m.values.items()])
        NLLMin = nllCalc.evalNLL(soln)
        fMin, tau1Min, tau2Min = soln

        print("Params for NLL minimum: ")
        print("------------------------------------------")
        print("F:\t" + str(fMin))
        print("Tau1:\t" + str(tau1Min))
        print("Tau2:\t" + str(tau2Min))
        print("")

        #for use later on in higher up methods
        self.minuit = m
        self.fitSoln = soln

    #prints simple errors as described by simple error method in report
    def simplisticErrors(self, nllCalc, soln):
        fMin, tau1Min, tau2Min = soln
        NLLMin = nllCalc.evalNLL(soln)

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

        #upperErrors
        fRoot = root.equalTo(nllCalc.evalNLL, shiftVal, soln, jumps, accuracys, [1,2])[0]
        tau1Root = root.equalTo(nllCalc.evalNLL, shiftVal, soln, jumps, accuracys, [0,2])[1]
        tau2Root = root.equalTo(nllCalc.evalNLL, shiftVal, soln, jumps, accuracys, [0,1])[2]

        fPosErr = fRoot - fMin
        tau1PosErr = tau1Root - tau1Min
        tau2PosErr = tau2Root - tau2Min

        #lowerErrors
        jumpsNeg = [-item for item in jumps]

        fRoot = root.equalTo(nllCalc.evalNLL, shiftVal, soln, jumpsNeg, accuracys, [1,2])[0]
        tau1Root = root.equalTo(nllCalc.evalNLL, shiftVal, soln, jumpsNeg, accuracys, [0,2])[1]
        tau2Root = root.equalTo(nllCalc.evalNLL, shiftVal, soln, jumpsNeg, accuracys, [0,1])[2]

        fNegErr = fMin - fRoot
        tau1NegErr = tau1Min - tau1Root
        tau2NegErr = tau2Min - tau2Root

        fMeanErr = 0.5*(fNegErr + fPosErr)
        tau1MeanErr = 0.5*(tau1NegErr + tau1PosErr)
        tau2MeanErr = 0.5*(tau2NegErr + tau2PosErr)

        print("Errors for NLL minimum (simplistic): ")
        print("------------------------------------------")
        print("F   :\t +" + str(fPosErr) + "\t-" +
            str(fNegErr) + "\t mean:" + str(fMeanErr))
        print("Tau1:\t +" + str(tau1PosErr) + "\t-" +
            str(tau1NegErr) + "\t mean:" + str(tau1MeanErr))
        print("Tau2:\t+" + str(tau2PosErr) + "\t-" +
            str(tau2NegErr) + "\t mean:" + str(tau2MeanErr))
        print("")

        #plotting error info_________________________________________
        numXVals = 50
        fRange = np.linspace(fMin - 2*fNegErr, fMin + 2*fPosErr, numXVals)
        tau1Range = np.linspace(tau1Min - 2*tau1NegErr, tau1Min + 2*tau1PosErr, numXVals)
        tau2Range = np.linspace(tau2Min - 2*tau2NegErr, tau2Min + 2*tau2PosErr, numXVals)

        fYvals = np.array([nllCalc.evalNLL(np.array([val, tau1Min, tau2Min])) for val in fRange])
        tau1Yvals = np.array([nllCalc.evalNLL(np.array([fMin, val, tau2Min])) for val in tau1Range])
        tau2Yvals = np.array([nllCalc.evalNLL(np.array([fMin, tau1Min, val])) for val in tau2Range])

        #centering around minimum
        fRangeCen = fRange - fMin
        tau1RangeCen = tau1Range - tau1Min
        tau2RangeCen = tau2Range - tau2Min

        fYvalsCen = fYvals - NLLMin
        tau1YvalsCen = tau1Yvals - NLLMin
        tau2YvalsCen = tau2Yvals - NLLMin

        #plots values of function around minimum
        plotter = Plot()
        plotter.errorInfo(fRangeCen, fYvalsCen, [fNegErr, fPosErr], 'F')
        plotter.errorInfo(tau1RangeCen, tau1YvalsCen, [tau1NegErr, tau1PosErr], r"$\tau_{1}$")
        plotter.errorInfo(tau2RangeCen, tau2YvalsCen, [tau2NegErr, tau2PosErr], r"$\tau_{2}$")
