'''
Returns the value of a function for a given set
of independent variables
'''
import numpy as np
import copy

class Function(object):

    #normalisation constant is becuase....
    #t in [0,10]
    #theta in [0, 2pi]
    def p1PDF(self, params):
        t, theta, tau1 = params

        norm1 = -3*np.pi * ( tau1*np.exp(-10./tau1) - tau1)
        val = (1. + (np.cos(theta))**2.)*np.exp(-t/tau1)
        return val/norm1


    def p2PDF(self, params):
        t, theta, tau2 = params

        norm2 = -3*np.pi * ( tau2*np.exp(-10./tau2) - tau2)
        val = 3*((np.sin(theta))**2.)*np.exp(-t/tau2)
        return val/norm2


    #overall pdf for a fraction
    #f of pdf1  and
    #(1-f) of pdf2
    def fPDF(self, params):
        f, t, theta, tau1, tau2 = params

        val = f*self.p1PDF([t, theta, tau1]) + (1-f)*self.p2PDF([t, theta, tau2])
        return val


class FixParams(object):
    def __init__(self, func, numOfParams, fixParamsVals, indexOfFixParams):
        self.func = func
        self.numOfParams = numOfParams
        self.fixParamsVals = fixParamsVals
        self.indexOfFixParams = indexOfFixParams

    def eval(self, freeParams):
        freeParamsCopy = copy.deepcopy(freeParams)
        fixParamsValsCopy = copy.deepcopy(self.fixParamsVals)

        finalParams = []

        for i in range(self.numOfParams):
            if i in self.indexOfFixParams:
                val = fixParamsValsCopy[0]
                fixParamsValsCopy.pop(0)
            else:
                val = freeParamsCopy[0]
                freeParamsCopy.pop(0)

            finalParams.append(val)

        return self.func(finalParams)


'''
Creates an instance of a class that is useful for evaluating a compsition
of 2 functions in Function (or in python libraries)
'''
class ComposeFunction(object):
    def __init__(self, funcToCompose, funcInitial):
        self.funcInitial = funcInitial
        self.funcToCompose = funcToCompose

    def evalCompose(self, params):
        return self.funcToCompose(self.funcInitial(params))

class AddFunction(object):
    def __init__(self, addX):
        self.addX = addX

    def evalAdd(self, a):
        return a + self.addX
