#================================================================================
#  Computer Simulation 4 with current-based synapses and no short-term plasticity
#    from  Legenstein, R., Pecevski, D. and Maass, W., A Learning Theory
#       for Reward-Modulated Spike-Timing-Dependent Plasticity with 
#       Application to Biofeedback 
# 
#  Author: Dejan Pecevski, dejan@igi.tugraz.at
#
#  Date: February 2008
#
#================================================================================

import sys
from sys import *
import os


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
from ReadoutModel import *
from TemplateInputModelKernelRwd import *


class PatternRewardSTDPExperiment(pcsim.Experiment):
        
    def defaultExpParameters(self):
        ep = self.expParams 
        
        # General simulation parameters        
        ep.DTsim = 1e-4
        
        ep.nTrainEpochs = 100
        ep.nTestEpochs = 10
        
        ep.trialT = 3
        
        # Network distribution parameters
        ep.netType = 'ST'
        ep.nThreads = 1
        ep.minDelay = 1e-3
        ep.maxDelay = 2   
        
        # Seeds of the experiment
        ep.numpyRandomSeed = 31342
        ep.pyRandomSeed = 124243        
        ep.constructionSeed = 3224356
        ep.simulationSeed = 134252439
        
        ep.runMode = "long"
        ep.input = "TemplateInputModelKernelRwd"
        
        ep.recordReadoutVm = True        
        ep.testWithNoise = True
        
        
        ep.numTrialsWithoutThreshold = 10
        ep.numTrialsRecordVm = 20
        
        
    def setupModels(self):        
        p = self.modelParams
        ep = self.expParams
        m = self.models
        net = self.net
        
        random.seed(ep.pyRandomSeed)
        numpy.random.seed(ep.numpyRandomSeed)
        
        
        m.input = eval(ep.input + '(self.net, self.expParams, p.get("input",{}))')        

        
        ep.Tsim = ep.nTrainEpochs * ep.trialT
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
        
        m.input.generate()
                
        # create the readout model
        m.readout = ReadoutModel(self.net, self.expParams, p.get("readout", {}), depModels = m.input.elements)        
        m.readout.generate()
        
        m.input.connectReadout(m.readout)

        
    def setupRecordings(self):
        r = self.recordings        
        r.input = self.models.input.setupRecordings()        
        r.readout = self.models.readout.setupRecordings()
        return r
    
    def simulate(self):
        ep = self.expParams        
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
        m = self.models
        
        currEpoch = 0  
        
        # Run simulation 
        print 'Running simulation:', datetime.today().strftime('%x %X')
        
        t0=datetime.today()
        
        self.net.reset();
        
        m.readout.setTestPhase()
        
        m.readout.increaseThreshold()
        
        m.readout.printSamplingTime()
        
        print "Test Before Learning:"
        while currEpoch < ep.nTestEpochs:
            if currEpoch % 10 == 0:
                stdout.write(str(currEpoch))
            else:
                stdout.write(".")
            m.input.reset(currEpoch)        
            self.net.advance(int(ep.trialT  / ep.DTsim))
            if ep.recordReadoutVm and currEpoch == ep.numTrialsRecordVm:
                m.readout.switchOffRecordVmReadout()
            if ep.recordReadoutVm and currEpoch == ep.numTrialsWithoutThreshold:    
                m.readout.setNormalThreshold()
            currEpoch += 1
        
        m.readout.setTrainPhase()
        
        print "Train Epoch: "
        while currEpoch < ep.nTrainEpochs + ep.nTestEpochs:
            if currEpoch % 10 == 0:
                stdout.write(str(currEpoch))
            else:
                stdout.write(".")
            m.input.reset(currEpoch)        
            self.net.advance(int(ep.trialT  / ep.DTsim))                                  
            currEpoch += 1
        
        m.readout.setTestPhase()
        
        print "Test Epoch: "
        while currEpoch < ep.nTrainEpochs + 2*ep.nTestEpochs:
            if currEpoch % 10 == 0:
                stdout.write(str(currEpoch))
            else:
                stdout.write(".")
            m.input.reset(currEpoch)
            self.net.advance(int(ep.trialT / ep.DTsim))
            if ep.recordReadoutVm and currEpoch == ep.nTrainEpochs + 2*ep.nTestEpochs - ep.numTrialsRecordVm:
                m.readout.switchOnRecordVmReadout()
            if ep.recordReadoutVm and currEpoch == ep.nTrainEpochs + 2*ep.nTestEpochs - ep.numTrialsWithoutThreshold:    
                m.readout.increaseThreshold()        
            currEpoch += 1
        
        t1=datetime.today()
        print 'Done.', (t1-t0).seconds, 'sec CPU time for', ep.Tsim, 's simulation time';        
        self.expParams.simDuration = (t1 - t0).seconds
        
        
    def scriptList(self):
        return ["PatternRewardSTDPExperiment.py"]
    
if __name__ == "__main__":
    
    numpySeedArray = [134381, 653434368, 2043436, 68688, 24213123, 61886]
    
    if len(sys.argv) > 1:
        runName = "final_"        
        seedNo = int(sys.argv[1])
        if len(sys.argv) > 2:
            directory = sys.argv[2]
        else:
            directory = "final_dir_" + datetime.today().strftime("%Y%m%d_%H%M%S")
        exper = PatternRewardSTDPExperiment('PatternRewardSTDP', 
                                            experParams = {"numpyRandomSeed" : numpySeedArray[seedNo], 
                                            "nTrainEpochs":1000, "nTestEpochs":60}, 
                                            modelParams = {"readout":{}}, 
                                            directory = directory)
        exper.run(runName+ "_" + sys.argv[1])
    else:
        exper = PatternRewardSTDPExperiment('PatternRewardSTDP', 
                                            experParams = {"numpyRandomSeed" : 4468, 
                                                           "nTrainEpochs":100, "nTestEpochs":30, "runMode" : "short",
                                                            "input" : "TemplateInputModelKernelRwd"} )
        exper.run("shortrun")
        