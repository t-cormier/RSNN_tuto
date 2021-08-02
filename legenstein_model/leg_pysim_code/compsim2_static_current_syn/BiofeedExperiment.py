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

 
from pypcsim import *
import pypcsimplus as pcsim

from numpy import *
import getopt
import numpy
from datetime import datetime
from math import *
from tables import *
from mpi4py import MPI

# models
from BiofeedModel import *
from PoissInputModel import *


class BiofeedExperiment(pcsim.Experiment):
        
    def defaultExpParameters(self):
        ep = self.expParams 
        
        # General simulation parameters
        ep.Tsim = 7200
        ep.DTsim = 1e-4
        
        # Network distribution parameters
        ep.netType = 'MT'
        ep.nThreads = 2
        ep.minDelay = 1e-3
        ep.maxDelay = 2   
        
        # Seeds of the experiment
        ep.numpyRandomSeed = 34234159
        ep.pyRandomSeed = 124243        
        ep.constructionSeed = 32241476
        ep.simulationSeed = 134212439
        
        ep.runMode = "long"        
        ep.modelName = "PoissInput"
        
    
    def setupModels(self):        
        p = self.modelParams
        ep = self.expParams
        random.seed(ep.pyRandomSeed)
        numpy.random.seed(ep.numpyRandomSeed)
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
                         
        self.models.input = eval(ep.modelName + '(self.net, self.expParams, p.get("input",{}))')        
        self.models.biofeed = Biofeed(self.net, self.expParams, p.get("biofeed",{}), depModels = self.models.input.elements)
        
        input_p = self.models.input.params
        biofeed_p = self.models.biofeed.params 
        
        self.models.input.generate()
        self.models.biofeed.generate()
        
        
    def setupRecordings(self):
        r = self.recordings        
        r.input = self.models.input.setupRecordings()
        r.biofeed = self.models.biofeed.setupRecordings()
        net = self.net
        m = self.models
                
        m.exc_ln = net.create(LinearNeuron(Rm = 1))
                
        for s in m.biofeed.elements.learning_plastic_syn:
            net.connect(s,m.exc_ln,StaticAnalogSynapse(delay = 1e-3))
        
        return r
    
    def simulate(self):
        ep = self.expParams
        biofeed_m = self.models.biofeed.elements
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
        
        # Run simulation 
        print 'Running simulation:';
        t0=datetime.today()
        
        self.net.add(SimProgressBar(Time.sec(ep.Tsim)), SimEngine.ID(0,0))
        
        print "Simulation start : " , datetime.today().strftime('%x %X')
        
        print "DTsim is ", ep.DTsim
        
        for s in biofeed_m.learning_plastic_syn:
            if self.net.object(s):                
                self.net.object(s).activeDASTDP = False
            
        for s in biofeed_m.learning_cond_plastic_syn:
            if self.net.object(s):
                self.net.object(s).activeDASTDP = False
        
        self.net.reset();
        self.net.advance(int(2 / ep.DTsim))
        
        for s in biofeed_m.learning_plastic_syn:
            if self.net.object(s):
                self.net.object(s).activeDASTDP = True
        
        
        for s in biofeed_m.learning_cond_plastic_syn:
            if self.net.object(s):
                self.net.object(s).activeDASTDP = True
        
        
        self.net.advance(int((ep.Tsim - 2) / ep.DTsim))
        
        t1=datetime.today()
        print 'Done.', (t1-t0).seconds, 'sec CPU time for', ep.Tsim, 's simulation time';        
        self.expParams.simDuration = (t1 - t0).seconds
        
        
    def scriptList(self):        
        return ["BiofeedExperiment.py"]
    
if __name__ == "__main__":    
    if len(sys.argv) > 1:
        numpyRandomSeed = [13048, 5012835, 656545, 25092385, 24086498]
        constructionSeed = [1650126, 5606836, 4509158, 63501348, 5023958]
        simulationSeed = [1045235, 65709388, 221230, 52065069, 5230598 ]
        pyRandomSeed = [10349, 643764370, 161374352, 16406098, 70605059]
        if len(sys.argv) > 2:        
            run_idx = int(sys.argv[2])
        else:
            run_idx = 3
            
        runName = "final_"
        if len(sys.argv) > 3:
            directory = sys.argv[3]
        else:
            directory = None
        exper = BiofeedExperiment('biofeed',
                                  experParams = { "numpyRandomSeed" : numpyRandomSeed[run_idx], 
                                                  "constructionSeed" : constructionSeed[run_idx], 
                                                  "simulationSeed" : simulationSeed[run_idx],
                                                  "pyRandomSeed" : pyRandomSeed[run_idx] }, 
                                                  modelParams = {}, 
                                                  directory = directory)        
        exper.run(runName)
    else:
        exper = BiofeedExperiment('biofeed', experParams = {"Tsim":200, "runMode" : "long"}, 
                                  modelParams = {})        
        exper.run("shortrun")
        