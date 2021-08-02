import os, sys
import matplotlib
matplotlib.use('Agg')


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
import LiquidModel400 as Liq
import SpeechStimulusInputModel as Inp
from traceback import *

class Liquid400Experiment(pcsim.Experiment):
            
    def defaultExpParameters(self):
        ep = self.expParams 
        
        # General simulation parameters
                
        ep.DTsim = 1e-4
        
        ep.trialT = 2.5
        
        # Network distribution parameters
        ep.netType = 'MT'
        ep.nThreads = 2
        ep.minDelay = 1e-3
        ep.maxDelay = 2   
        
        # Seeds of the experiment
        ep.numpyRandomSeed = 31342
        ep.pyRandomSeed = 124243        
        ep.constructionSeed = 3224356
        ep.simulationSeed = 134252439
        
        ep.runMode = "long"        
        ep.liquid = "Liq.LiquidModel400"        
        ep.input = "Inp.SpeechStimulusInputModel"  
        
        ep.ipEngineID = 0
        
    def setupModels(self):        
        p = self.modelParams
        ep = self.expParams
        m = self.models
        net = self.net
        ep.Tsim = ep.trialT        
        
        random.seed(ep.pyRandomSeed)        
        numpy.random.seed(int(ep.numpyRandomSeed))
                                     
        m.liquid = eval(ep.liquid + '(self.net, self.expParams, p.get("liquid",{}))')
                
        m.input = eval(ep.input + '(self.net, self.expParams, p.get("input",{}), depModels = m.liquid)')        
        
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
        
        m.liquid.generate()                    
        m.input.generate()
                    
    def reset(self):
        self.net.reset()
        
    def setupRecordings(self):
        r = self.recordings        
        r.input = self.models.input.setupRecordings()
        r.liquid = self.models.liquid.setupRecordings()    
        return r

    def getOutput(self):
        out = pcsim.Dictionary()
        r = self.recordings
        net = self.net
        out.exc_spikes = [ array(r.liquid.exc_spikes.object(i).getRecordedValues()) for i in range(r.liquid.exc_spikes.size()) ]
        out.inh_spikes = [ array(r.liquid.inh_spikes.object(i).getRecordedValues()) for i in range(r.liquid.inh_spikes.size()) ]
        out.numSynapses = [r.liquid.numSynapses]
        
        input = [ array(r.input.input.object(i).getRecordedValues()) for i in range(r.input.input.size()) ]
        return out,input
        
    
    def simulate(self):
        ep = self.expParams
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
        ep.Tsim = ep.trialT
        
        t0=datetime.today()
        self.net.advance(int(ep.Tsim/ep.DTsim))                            
        t1=datetime.today()
        print 'Done.', (t1-t0).seconds, 'sec CPU time for', ep.Tsim, 's simulation time';        
        self.expParams.simDuration = (t1 - t0).seconds
        
        
    def scriptList(self):
        return ["Liquid400Experiment.py"]
 
if __name__ == "__main__":
    exper = Liquid400Experiment('test', experParams = {})
    exper.setup()
    exper.models.input.resetStimulus((1,1,1))
    exper.run('shortrun', saveData = False)
    print exper.getOutput()