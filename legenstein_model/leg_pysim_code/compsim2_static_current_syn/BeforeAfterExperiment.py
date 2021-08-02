#==================================================================================
#  Computer simulation 2 - with current based synapses and no short-term plasticity
#       from Legenstein, R., Pecevski, D. and Maass, W., A Learning  Theory 
#       for Reward-Modulated Spike-Timing-Dependent Plasticity with Application 
#       to Biofeedback
# 
#  Author: Dejan Pecevski, dejan@igi.tugraz.at
#
#  Date: February 2008
#===================================================================================

import sys
import os

sys.path.append("..")


from pypcsim import *
import pypcsimplus as pcsim
from numpy import *
import random, getopt
import numpy
from datetime import datetime
from math import *
from tables import *
from math import exp
from mpi4py import MPI
from BeforeAfterBiofeedModel import *
from PoissInputModel import *

class BeforeAfterExperiment(pcsim.Experiment):
        
    def defaultExpParameters(self):
        ep = self.expParams 
        
        # General simulation parameters
        ep.Tsim = 20
        ep.DTsim = 1e-4
        
        # Network distribution parameters
        ep.netType = 'ST'
        ep.nThreads = 1
        ep.minDelay = 1e-3
        ep.maxDelay = 2   
        
        # Seeds of the experiment
        ep.numpyRandomSeed = 153564312
        ep.pyRandomSeed = 1615335    
        ep.constructionSeed = 31653476
        ep.simulationSeed = 13421639
        
        ep.runMode = "long"        
        ep.modelName = "PoissInput"
        
    
    def setupModels(self):        
        p = self.modelParams
        ep = self.expParams
        random.seed(ep.pyRandomSeed)
        numpy.random.seed(ep.numpyRandomSeed)
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
                         
        self.models.input = eval(ep.modelName + '(self.net, self.expParams, p.get("input",{}))')        
        self.models.biofeed = BeforeAfterBiofeedModel(self.net, self.expParams, p.get("biofeed",{}), depModels = self.models.input.elements)

        input_p = self.models.input.params
        biofeed_p = self.models.biofeed.params 
        
        self.models.input.generate()
        self.models.biofeed.generate()
        
        
    def setupRecordings(self):
        r = self.recordings
        r.biofeed = self.models.biofeed.setupRecordings()        
        return r
        
    def scriptList(self):
        return ["BeforeAfterExperiment.py"]
    
if __name__ == "__main__":    
        exper = BeforeAfterExperiment('beforeAfter', experParams = {}, modelParams = {})
        exper.run("longrun")
