import matplotlib.pylab as pl
import numpy as np

class Plot(object):

    def plotDistributions(self, t_thetaVals, title):
        tVals = [item[0] for item in t_thetaVals]
        thetaVals = [item[1] for item in t_thetaVals]

        pl.figure(1)
        pl.scatter(thetaVals, tVals, s=1)
        pl.title( title + r"$\theta$ vs $t$ occurence scatter")
        pl.xlabel(r"$t$")
        pl.ylabel(r"$\theta$")

        #plotting individually
        tBins = 70
        pl.figure(2)
        pl.hist(tVals, bins=tBins)
        pl.title( title + r"$t$ distribution")
        pl.xlabel(r"$t$")
        pl.ylabel("frequency")

        thetaBins = 70
        pl.figure(3)
        pl.hist(thetaVals, bins=thetaBins)
        pl.title( title + r"$\theta$ distribution")
        pl.xlabel(r"$\theta$")
        pl.ylabel("frequency")

        pl.show()

    def errorInfo(self, xRange, yVals, errors, label):
        numXVals = 50
        lowerError, upperError = errors

        pl.title(r"$\Delta$" + label + r" vs. $\Delta$NLL")
        pl.xlabel(r"$\Delta$" + label)
        pl.ylabel("$\Delta NLL$")
        pl.axvline(x = errors[0], linestyle=':', color='r')
        pl.axvline(x = errors[1], linestyle=':', color='r')
        pl.axhline(y = 0.5, linewidth=0.5, linestyle='--', color='dimgray')
        pl.plot(xRange, yVals)
        pl.show()
