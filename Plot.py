import matplotlib.pylab as pl
import numpy as np

class Plot(object):

    def plotDistributions(self, t_thetaVals, title):
        tVals = [item[0] for item in t_thetaVals]
        thetaVals = [item[1] for item in t_thetaVals]

        pl.figure(1)
        pl.scatter(thetaVals, tVals, s=1)
        pl.title( title + r" $\theta$ vs $t$ occurence scatter")
        pl.xlabel(r"$\theta$")
        pl.ylabel(r"$t$")

        #plotting individually
        tBins = 70
        pl.figure(2)
        pl.hist(tVals, bins=tBins)
        pl.title( title + r" $t$ distribution")
        pl.xlabel(r"$t$")
        pl.ylabel("frequency")

        thetaBins = 70
        pl.figure(3)
        pl.hist(thetaVals, bins=thetaBins)
        pl.title( title + r" $\theta$ distribution")
        pl.xlabel(r"$\theta$")
        pl.ylabel("frequency")

        pl.show()

    def errorCtr(self, minuit, soln):
        numPoints = 80
        sigma = 1.0

        minuit.set_errordef(0.5)

        f_tau1Ctr1 = minuit.mncontour('f','tau1', numpoints=numPoints, sigma=sigma)[2]
        f_tau1x1 = [val[0] for val in f_tau1Ctr1 ]
        f_tau1y1 = [val[1] for val in f_tau1Ctr1 ]

        print("done 1/6")

        f_tau2Ctr1 = minuit.mncontour('f','tau2', numpoints=numPoints, sigma=sigma)[2]
        f_tau2x1 = [val[0] for val in f_tau2Ctr1 ]
        f_tau2y1 = [val[1] for val in f_tau2Ctr1 ]

        print("done 2/6")

        tau2_tau1Ctr1 = minuit.mncontour('tau2','tau1', numpoints=numPoints, sigma=sigma)[2]
        tau2_tau1x1 = [val[0] for val in tau2_tau1Ctr1 ]
        tau2_tau1y1 = [val[1] for val in tau2_tau1Ctr1 ]

        print("done 3/6")

        minuit.set_errordef(1.0)

        f_tau1Ctr2 = minuit.mncontour('f','tau1', numpoints=numPoints, sigma=sigma)[2]
        f_tau1x2 = [val[0] for val in f_tau1Ctr2 ]
        f_tau1y2 = [val[1] for val in f_tau1Ctr2 ]

        print("done 4/6")

        f_tau2Ctr2 = minuit.mncontour('f','tau2', numpoints=numPoints, sigma=sigma)[2]
        f_tau2x2 = [val[0] for val in f_tau2Ctr2 ]
        f_tau2y2 = [val[1] for val in f_tau2Ctr2 ]

        print("done 5/6")

        tau2_tau1Ctr2 = minuit.mncontour('tau2','tau1', numpoints=numPoints, sigma=sigma)[2]
        tau2_tau1x2 = [val[0] for val in tau2_tau1Ctr2 ]
        tau2_tau1y2 = [val[1] for val in tau2_tau1Ctr2 ]

        print("done 6/6")

        pl.figure(1)
        pl.title(r"f vs $\tau_{1}$ error contour")
        pl.xlabel("f")
        pl.ylabel(r"$\tau_{1}$")
        pl.plot(f_tau1x1, f_tau1y1, '-', label='0.5')
        pl.plot(f_tau1x2, f_tau1y2, '-', label='1.0')
        pl.scatter([soln[0]],[soln[1]])
        pl.legend(loc='best')
        pl.savefig("data/results/part6/2/f_tau1.png")

        pl.figure(2)
        pl.title(r"f vs $\tau_{2}$ error contour")
        pl.xlabel("f")
        pl.ylabel(r"$\tau_{2}$")
        pl.plot(f_tau2x1, f_tau2y1, '-', label='0.5')
        pl.plot(f_tau2x2, f_tau2y2, '-', label='1.0')
        pl.scatter([soln[0]],[soln[2]])
        pl.legend(loc='best')
        pl.savefig("data/results/part6/2/f_tau2.png")

        pl.figure(3)
        pl.title(r"$\tau_{2}$ vs $\tau_{1}$ error contour")
        pl.xlabel(r"$\tau_{2}$")
        pl.ylabel(r"$\tau_{1}$")
        pl.plot(tau2_tau1x1, tau2_tau1y1, '-', label='0.5')
        pl.plot(tau2_tau1x2, tau2_tau1y2, '-', label='1.0')
        pl.scatter([soln[2]],[soln[1]])
        pl.legend(loc='best')
        pl.savefig("data/results/part6/2/tau2_tau1.png")

    def errorInfo(self, xRange, yVals, errors, label):
        numXVals = 50
        lowerError, upperError = errors

        pl.title(r"$\Delta$" + label + r" vs. $\Delta$NLL")
        pl.xlabel(r"$\Delta$" + label)
        pl.ylabel("$\Delta NLL$")
        pl.axvline(x = -lowerError, linestyle=':', color='r')
        pl.axvline(x = upperError, linestyle=':', color='r')
        pl.axhline(y = 0.5, linewidth=0.5, linestyle='--', color='dimgray')
        pl.plot(xRange, yVals)
        pl.show()
