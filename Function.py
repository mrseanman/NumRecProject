#File for classes related to Function.
#Contains Function itself and classes for 'modulating'
#instances and methods in Function

import numpy as np
import copy

'''
The methods of this class all return evaluations of PDFs relating to the
report.

As a rule, the functions take all their parameters bundled in an array.
This makes for a more uniform interface between these functions and
other methods that interact with them.

Most importantly it allows for outside methods to deal with functions of
that require different numbers of parameters cleanly.

The methods in this class are dumb, more useful things are done with
them using the other classes in this file.
'''
class Function(object):

    #P1___________________________________________________
    #this assumes t in [0,infinity]
    def p1PDF(self, params):
        t, theta, tau1 = params
        norm1 = 3*np.pi * tau1
        val = (1. + (np.cos(theta))**2.)*np.exp(-t/tau1)
        return val/norm1

    #same as above but a different normalisation constant because
    #t in [0,10]
    #theta in [0, 2pi]
    def p1PDF_tRange(self, params):
        t, theta, tau1 = params

        norm1 = -3*np.pi * ( tau1*np.exp(-10./tau1) - tau1)
        val = (1. + (np.cos(theta))**2.)*np.exp(-t/tau1)
        return val/norm1

    #P2___________________________________________________
    #this assumes t in [0,infinity]
    def p2PDF(self, params):
        t, theta, tau2 = params

        norm2 = 3*np.pi * tau2
        val = 3*((np.sin(theta))**2.)*np.exp(-t/tau2)
        return val/norm2

    #same as above but a different normalisation constant because
    #t in [0,10]
    #theta in [0, 2pi]
    def p2PDF_tRange(self, params):
        t, theta, tau2 = params

        norm2 = -3*np.pi * ( tau2*np.exp(-10./tau2) - tau2)
        val = 3*((np.sin(theta))**2.)*np.exp(-t/tau2)
        return val/norm2

    #PFull________________________________________________
    #Full PDF P
    #f * pdf1  + (1-f)*pdf2
    def fPDF(self, params):
        f, t, theta, tau1, tau2 = params

        val = f*self.p1PDF([t, theta, tau1]) + (1-f)*self.p2PDF([t, theta, tau2])
        return val

    #equivalent of fPDF for when
    #t in [0,10]
    def fPDF_tRange(self,params):
        f, t, theta, tau1, tau2 = params
        val = f*self.p1PDF_tRange([t, theta, tau1]) + (1-f)*self.p2PDF_tRange([t, theta, tau2])
        return val

    #P_Theta_Independent__________________________________
    #P1 and P2 equivalent for no theta dependence
    # = P1 and P2 integrated over all theta [0,2pi]
    #only one because they both end up the same
    def thetaIndepPDF(self, params):
        t, tau = params
        norm = tau
        val = np.exp(-t/tau)
        return val/norm

    #equivalent of fPDF for no theta dependence
    def fThetaIndepPDF(self, params):
        f, t, tau1, tau2 = params
        val = f*self.thetaIndepPDF([t,tau1]) + (1-f)*self.thetaIndepPDF([t,tau2])
        return val



'''
An instance of this class is used when we want to reduce the number
of parameters for a method in Function.

A use case of this is when certain parameters are fixed by data
and we want our function to only need the remaining free parameters.

e.g. function.fPDF requires 5 values in params.
t and theta are fixed by a data point.
FixParams.eval now only requires 3 values in freeParams.

This might seem like a strange workaround, but it allows for
the minimisers etc. to be as dumb and as general as possible.

In principal this can be used with methods not in Function
'''
class FixParams(object):
    def __init__(self, func, numOfParams, fixParamsVals, indexOfFixParams):
        self.func = func
        self.numOfParams = numOfParams
        self.fixParamsVals = fixParamsVals
        self.indexOfFixParams = indexOfFixParams
        self.indexOfFreeParams = [index for index in range(self.numOfParams) if not(index in indexOfFixParams)]

    def eval(self, freeParams):
        params = range(self.numOfParams)
        for index, val in enumerate(self.fixParamsVals):
            params[self.indexOfFixParams[index]] = val
        for index, val in enumerate(freeParams):
            params[self.indexOfFreeParams[index]] = val
        return self.func(params)


'''
Modulates instances of Function

An instance of this class is used to compose two functions and create
a new method evalCompose that requires only one set of parameters
and to python just looks like a normal method.

This might seem like a strange workaround but when we send a function
to a minimiser, there is no easy way to give the minimiser/root finder
a composition of functions.

A use case is when we want to find where a function f(params) = val.
This is equivalent to finding the root of f(params) - val.
So we compose (lambda x: x-val) with f(params) and then find the
root of FixParams.eval()

This is to make everything else that interacts with functions
as dumb and as general as possible.
'''
class ComposeFunction(object):
    def __init__(self, funcToCompose, funcInitial):
        self.funcInitial = funcInitial
        self.funcToCompose = funcToCompose

    def evalCompose(self, params):
        return self.funcToCompose(self.funcInitial(params))
