import matplotlib
matplotlib.use('Agg')
import sys
import pypcsimplus as pcsim
sys.path.append('../packages')
from pyV1.dataformats import *


class StimulusModel(pcsim.Model):
    
    def defaultParameters(self):
        p = self.params
        p.nInputNeurons = 86
   
    def __init__(self, net, experParams, new_params = {}, depModels = pcsim.Dictionary(), stimulus = None):
        pcsim.Model.__init__(self, net, experParams, new_params, depModels)         
        self.stimulus = stimulus
        
        
    def setStimulus(self,stimulus):
        self.stimulus = stimulus
        
    def reset(self):
        p = self.params
        m = self.elements
        ep = self.expParams
        for i in range(p.nInputNeurons):
            if m.popul.object(i):
                m.popul.object(i).reset(ep.DTsim)

    def generate(self):
        p = self.params
        net = self.net
        m = self.elements
        
        if not hasattr(m,'popul'):
            print "population created in input"
            m.popul = pcsim.SimObjectPopulation(net, pcsim.SpikingInputNeuron(), p.nInputNeurons)
        
        if self.stimulus:
            print "stimulus detected and filling population with stimulus"
            for i in range(len(self.stimulus.channel)):
                m.popul.object(i).setSpikes(self.stimulus.channel[i].data)
        
        return self.elements
    
    def scriptList(self):
        return ["StimulusModel.py"]

    