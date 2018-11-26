import copy
import math
import numpy as np
from Function import Function
from iminuit import Minuit

'''
Class for finding minima and roots using simple methods.

Philosphy was to make the methods as general as was reasonable.
So min and root can work with function with any number of numerical
parameters. So long as the function takes the parameters
all bundled in a list.

From a deep fear of mutable lists, there is a lot of copy.deepcopy
This might be lazy but it allows for more certainty and understanding
of the methods behaviour.

In general these methods are kept clean and dumb. Their utility is
more in their understandability rather than their efficiency
or utility, since there are other libraries that do that better anyway.

There is no protection against taking too many iterations.
Try to minimise x^3 and it will stall. God bless ctrl-c

No info about the locality or globality of minima is taken in to account.
'''
class Optimise(object):

    #minimising_____________________________________________________________
    #not actually used in report. Opted for minuit instead
    #But copied from previous work so kept for good measure.
    #
    #returns values of params at function (local) minimum
    #
    #func is function to be minimised only argument must be list params
    #fParamsGuess is list of param starting positions
    #paramsJump is list of initial deltas for parameters
    #paramsAccuracy is list of accuracys on each parameter for minimum
    #paramsToFix is list of indexes of params to... keep fixed
    #
    #Minimises a function in a very simple and dumb way.
    #all other parameters are fixed while one parameter is minimised
    #returns values of parameters at minimum
    def min(self, func, fParamsGuess, paramsJump,
     paramsAccuracy, paramsToFix):
        params = copy.deepcopy(fParamsGuess)
        numParams = len(params)

        #how many times to do the whole minimisation
        totalRepeats = 10

        for i in range(totalRepeats):
            #goes through all the parameters
            #minimises function by minimising each param
            #one at a time while fixing all others
            for paramIndex in range(numParams ):
                if not(paramIndex in paramsToFix):
                    params = self.minSingleParam(func, paramIndex, params, paramsJump[paramIndex], paramsAccuracy[paramIndex])
        return params

    #returns value of parameter at a minimum along the line
    #
    #scans variable for when it passes a minimum.
    #once it passes the jump reverses and is divided by two
    #certain number of passes are done until accuracy is reached
    def minSingleParam(self, func, freeParamIndex,
        fParams, freeParamJump, accuracy):
        params = copy.deepcopy(fParams)

        numIter = int(math.ceil(math.log(freeParamJump/accuracy, 2.)))
        if numIter < 1:
            numIter = 1

        #finds initial direction
        val0 = func(params)
        params[freeParamIndex] += freeParamJump
        val1 = func(params)
        direction = np.sign(val0 - val1)

        val1 = val0
        for i in range(numIter):
            changeDir = False
            while not(changeDir):
                val0 = val1
                params[freeParamIndex] += direction*freeParamJump
                val1 = func(params)
                if val1 > val0:
                    changeDir = True
            #halves the jump at each direction change
            freeParamJump /= 2.
            direction *= -1.
        return params

    #root_finding____________________________________________________________
    #returns values of parameters at root
    #i.e. finds params for when func(params)=0
    #paramsToFix is list of parameter indexes that should be fixed
    #
    #arguments are same as for min
    #
    #Works in a simple and dumb way.
    #all other parameters are kept fixed while one parameter is moved around
    #to find either a root or a minima along that line.
    #Then do the same for all the other parameters
    def root(self, func, fParamsGuess,
        paramsJump, paramsAccuracy, paramsToFix):
        params = copy.deepcopy(fParamsGuess)
        numParams = len(params)
        for paramIndex in range(numParams):
            if not(paramIndex in paramsToFix):
                params = self.rootSingleParam(func, paramIndex, params, paramsJump[paramIndex], paramsAccuracy[paramIndex])
        return params

    #retruns value of paramers at minimum or root along the line.
    #But only value in freeParamIndex is changed
    def rootSingleParam(self, func, freeParamIndex,
        fParams, freeParamJump, accuracy):
        params = copy.deepcopy(fParams)

        numIter = int(math.ceil(-math.log( abs(accuracy/freeParamJump), 2.)))
        if numIter < 1:
            numIter = 1

        #finds initial direction
        val0 = func(params)
        params[freeParamIndex] += freeParamJump
        val1 = func(params)
        direction = -(np.sign(val1 - val0) * np.sign(val0))
        val1 = val0

        for i in range(numIter):
            val0 = val1
            changeDir = False
            while ((np.sign(val0) == np.sign(val1)) and not(changeDir)):
                val0 = val1
                params[freeParamIndex] += direction*freeParamJump
                val1 = func(params)
                if (abs(val1) - abs(val0) > 0.) and (np.sign(val1) == np.sign(val0)):
                    changeDir = True

            if changeDir:
                print(freeParamIndex)
                print("Changed Direction while finding root!")

            freeParamJump /= 2.
            direction *= -1
        return params

    #returns params where func = x
    #i.e. finds root of func - x
    def equalTo(self, func, x, fParamsGuess,
        paramsJump, paramsAccuracy, paramsToFix):
        paramsGuess = copy.deepcopy(fParamsGuess)
        self.func = func
        self.valueToFind = x
        root = self.root(self.funcMinus, paramsGuess, paramsJump, paramsAccuracy, paramsToFix)
        return root

    def funcMinus(self, params):
        return self.func(params) - self.valueToFind

    #error_finding___________________________________________________________
    #returns full error on all the parameter minimums as described in the
    #full error method in the report.
    #
    #func is function in question
    #minParams is list of values of params at func minimum.
    #errorDef for NLL is 0.5
    #jumps is list of starting deltas for scan
    #accuracys is list of accuracys for errors on params
    #limits is tuple of duple (lowerLim, upperLim) for minimiser scan.
    #   i.e. ((param1Lower, param1Upper), (param2Lower, param2Upper)...)
    #names is tuple of str to call each parameter
    #
    #Positive jumps will give positive error and visa versa
    #Finds where func = valAtMin + errorDef by scanning each
    #param all while constantly minimising func and updating
    #the parameters that are not being scanned accordingly.
    def error(self, func, minParams, errorDef,
        jumps, accuracys, limits, names):
        valAtMin = func(minParams)
        valAtError = valAtMin + errorDef

        errors = []
        for index in range(len(minParams)):
            paramError = self.errorSingleParam(valAtError, func, index,
                minParams, jumps[index], accuracys[index], limits, names)

            errors.append(paramError)
            print("Done: " + str(index) + "/" + str())

        return errors

    #returns param_at_error - param_at_minimum
    #
    #valAtError is same as errorDef for error
    #funcInit is function in question
    #fParams is list of values of parameters at minimum of funcInit
    #rest of parameters are the same as for error
    #
    #very similar method to rootSingleParam but after jumping the free param
    #the remaining params are changed while fixing the free param so as to
    #minimise func as best as possible.
    def errorSingleParam(self, valAtError, funcInit,
     freeParamIndex, fParams, freeParamJump, accuracy, limits, names):
        #all this because we are finding root of func-valAtError
        self.valueToFind = valAtError
        self.func = funcInit
        func = self.funcMinus

        params = copy.deepcopy(fParams)

        #creates a well formatted tuple of which vars to fix
        fix = tuple([index==freeParamIndex for index in range(len(params))])

        fitArgs = dict(fix=fix,
                        limit=limits, name = tuple(names),
                        errordef=1, print_level=0)
        m = Minuit.from_array_func(func, tuple(params), **fitArgs)

        numIter = int(math.ceil(-math.log( abs(accuracy/freeParamJump), 2.)))
        if numIter < 1:
            numIter = 1

        #finds initial direction
        val0 = func(params)
        params[freeParamIndex] += freeParamJump
        val1 = func(params)
        direction = -(np.sign(val1 - val0) * np.sign(val0))

        for i in range(numIter):
            val0 = val1
            changeDir = False
            while ((np.sign(val0) == np.sign(val1)) and not(changeDir)):
                val0 = val1
                params[freeParamIndex] += direction*freeParamJump

                #forming new initial conditions
                for i in range(len(params)):
                    m.values[i] = params[i]

                m.migrad()

                #extracting new params
                for i in range(len(params)):
                    params[i] = m.values[i]

                #m.fval is value of func at last computed minimum
                val1 = m.fval

                if (abs(val1) - abs(val0) > 0.) and (np.sign(val1) == np.sign(val0)):
                    changeDir = True

            if changeDir:
                print(freeParamIndex)
                print("Changed Direction while finding root!")

            freeParamJump /= 2.
            direction *= -1

        return params[freeParamIndex] - fParams[freeParamIndex]
