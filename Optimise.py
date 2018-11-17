'''
Class to minimiseChior find roots of given function
'''

#test for git
import copy
import math
import numpy as np
from Function import Function, ComposeFunction

class Optimise(object):

    def min(self, func, fParamsGuess, paramsJump, paramsAccuracy, paramsToFix):
        params = copy.deepcopy(fParamsGuess)
        numParams = len(params)

        totalRepeats = 10
        for i in range(totalRepeats):
            #goes through all the parameters
            #minimises function by minimising each param
            #one at a time while fixing all others
            for paramIndex in range(numParams ):
                if not(paramIndex in paramsToFix):
                    params = self.minSingleParam(func, paramIndex, params, paramsJump[paramIndex], paramsAccuracy[paramIndex])
        return params

    def minSingleParam(self, func, freeParamIndex, fParams, freeParamJump, accuracy):
        params = copy.deepcopy(fParams)

        numIter = int(math.ceil(-math.log(accuracy/freeParamJump, 2.)))
        if numIter < 1:
            numIter = 1

        #finds initial direction
        val0 = func(params)
        params[freeParamIndex] += freeParamJump
        val1 = func(params)
        if val1 <= val0:
            direction = 1.
        else:
            direction = -1.

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


    def root(self, func, fParamsGuess, paramsJump, paramsAccuracy, paramsToFix):
        params = copy.deepcopy(fParamsGuess)
        numParams = len(params)
        for paramIndex in range(numParams):
            if not(paramIndex in paramsToFix):
                params = self.rootSingleParam(func, paramIndex, params, paramsJump[paramIndex], paramsAccuracy[paramIndex])
        return params


    def rootSingleParam(self, func, freeParamIndex, fParams, freeParamJump, accuracy):
        params = copy.deepcopy(fParams)

        numIter = int(math.ceil(-math.log( abs(accuracy/freeParamJump), 2.)))
        if numIter < 1:
            numIter = 1

        #finds initial direction
        val0 = func(params)
        params[freeParamIndex] += freeParamJump
        val1 = func(params)
        if (np.sign(val1 - val0) * np.sign(val0)) == -1 :
            direction = 1.
        else:
            direction = -1.
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
                print("Changed Direction while finding root\nChoose better values!\n")

            freeParamJump /= 2.
            direction *= -1
        return params

    #returns where func = x
    #i.e. finds root of func - x
    def equalTo(self, x, func, fParamsGuess, paramsJump, paramsAccuracy, paramsToFix):
        paramsGuess = copy.deepcopy(fParamsGuess)
        self.func = func
        self.valueToFind = x
        root = self.root(self.funcMinus, paramsGuess, paramsJump, paramsAccuracy, paramsToFix)
        return root


    def funcMinus(self, params):
        return self.func(params) - self.valueToFind
