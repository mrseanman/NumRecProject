'''
Class to wrap up data from a file
'''

class Data(object):

    #dataset points to the data file
    def __init__(self, dataFileName):
        self.importData(dataFileName)

    def importData(self, dataFileName):
        self.tVals = []
        self.thetaVals= []

        dataText = open(dataFileName, "r")
        lines = dataText.readlines()

        #number of datapoints
        self.numData = len(lines)

        for line in lines:
            vals = line.split()
            self.tVals.append(float(vals[0]))
            self.thetaVals.append(float(vals[1]))

        dataText.close()
