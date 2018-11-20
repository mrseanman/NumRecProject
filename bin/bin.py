def fitTVals(self):
    filename = "data/datafile-Xdecay.txt"
    data = Data(filename)
    func = Function()
    dataForNLL = [[item] for item in data.tVals]

    #too much data!! actual had 100K
    #---------------
    partialData = []
    for i in range(100000):
        partialData.append(dataForNLL[i])

    #---------------

    NLL = NLL(func.fThetaIndepPDF, partialData, [1], 4)

    #initial guesses
    fInitial = 0.4
    tau1Initial = 1.9
    tau2Initial = 1.2
    initialGuess = [fInitial, tau1Initial, tau2Initial]

    #bounds
    fBound = (0.00001,0.999999)
    tau1Bound = (0.00001,20.)
    tau2Bound = (0.00001,20.)
    bounds = [fBound, tau1Bound, tau2Bound]

    tolerance = 0.00000001
    soln = optimize.minimize(NLL.evalNLL, initialGuess, bounds=bounds, options={'disp': False}).x
    fMin, tau1Min, tau2Min = soln
    NLLMin = NLL.evalNLL(soln)

    print("Vals for NLL minimum: ")
    print("------------------------------------------")
    print("F:\t" + str(fMin))
    print("Tau1:\t" + str(tau1Min))
    print("Tau2:\t" + str(tau2Min))
    print("")

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
    shift = lambda a: a - shiftVal
    rootShift = ComposeFunction(shift, NLL.evalNLL)

    #upperErrors
    fRoot = root.root(rootShift.evalCompose, list(soln), jumps, accuracys, [1,2])[0]
    tau1Root = root.root(rootShift.evalCompose, list(soln), jumps, accuracys, [0,2])[1]
    tau2Root = root.root(rootShift.evalCompose, list(soln), jumps, accuracys, [0,1])[2]

    fPosErr = fRoot - fMin
    tau1PosErr = tau1Root - tau1Min
    tau2PosErr = tau2Root - tau2Min

    #lowerErrors
    jumpsNeg = [-item for item in jumps]

    fRoot = root.root(rootShift.evalCompose, list(soln), jumpsNeg, accuracys, [1,2])[0]
    tau1Root = root.root(rootShift.evalCompose, list(soln), jumpsNeg, accuracys, [0,2])[1]
    tau2Root = root.root(rootShift.evalCompose, list(soln), jumpsNeg, accuracys, [0,1])[2]

    fNegErr = fMin - fRoot
    tau1NegErr = tau1Min - tau1Root
    tau2NegErr = tau2Min - tau2Root

    fMeanErr = 0.5*(fNegErr + fPosErr)
    tau1MeanErr = 0.5*(tau1NegErr + tau1PosErr)
    tau2MeanErr = 0.5*(tau2NegErr + tau2PosErr)

    print("Errors NLL minimum: ")
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

    fYvals = [NLL.evalNLL([val, tau1Min, tau2Min]) for val in fRange]
    tau1Yvals = [NLL.evalNLL([fMin, val, tau2Min]) for val in tau1Range]
    tau2Yvals = [NLL.evalNLL([fMin, tau1Min, val]) for val in tau2Range]

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








    soln = optimize.minimize(nllCalc.evalNLL, initialGuess, bounds=bounds, options={'disp': False}).x
