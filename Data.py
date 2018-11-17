'''
Class to wrap up data from a file
'''

class Data(object):

    #dataset points to the data file
    def __init__(self, dataFileName):
        self.importData(dataFileName)


    #puts data in dataFileName in relevant vals or errVals
    #this is for dealing with tab separated values as in the checkpoint
    def importData(self, dataFileName):
        self.data = [[],[]]
        self.errData = []

        dataText = open(dataFileName, "r")
        lines = dataText.readlines()

        #number of datapoints
        self.numData = len(lines)

        for line in lines:
            vals = line.split()
            numOfVals = len(vals)

            self.data[0].append(float(vals[0]))
            if numOfVals > 1:
                self.data[1].append(float(vals[1]))
                if numOfVals > 2:
                    self.errData.append(float(vals[2]))

        dataText.close()
