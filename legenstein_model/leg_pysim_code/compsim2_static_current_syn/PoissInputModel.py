from pypcsimplus import *
import pypcsimplus as pcsim
from numpy import *

class PoissInput(pcsim.Model):
    
    def defaultParameters(self):
        p = self.params
        
        # input parameters                
        p.inputRate = 13.5
        p.nInputNeurons = 190
        p.nExcNeurons = 150
        p.Trefract = 5e-4
        p.input_type = 'inputNeurons'
        
    
    def setupRecordings(self):
        p = self.params
        r = Recordings(self.net)
        if p.input_type != 'inputNeurons':
            r.spikes = self.elements.input_nrn_popul.record(SpikeTimeRecorder())
        else:
            r.spikes = self.elements.inputs        
        return r
    
    
    def generate(self):        
        m = self.elements
        dm = self.depModels
        p = self.params
        ep = self.expParams
            
        if p.input_type == 'inputNeurons':
            m.input_nrn_popul = SimObjectPopulation(self.net, SpikingInputNeuron(), p.nInputNeurons)
         
            m.inputs = sort(random.uniform(0, ep.Tsim - int(p.inputRate * ep.Tsim) * p.Trefract, m.input_nrn_popul.size() * int(p.inputRate * ep.Tsim)).reshape(m.input_nrn_popul.size(), 
                          int(p.inputRate * ep.Tsim)), 1) + outer(ones(m.input_nrn_popul.size()), cumsum(ones(int(p.inputRate * ep.Tsim))*p.Trefract))
        
            for i in range(m.input_nrn_popul.size()):
                if m.input_nrn_popul.object(i):
                    m.input_nrn_popul.object(i).setSpikes(m.inputs[i])
        else:
            m.input_nrn_popul = SimObjectPopulation( self.net, LinearPoissonNeuron(1, 1, 5e-4, 0, p.inputRate), p.nInputNeurons )
    
        exc_nrn_ids = []
        inh_nrn_ids = []    
        for i in range(0,p.nExcNeurons):
            exc_nrn_ids.append(m.input_nrn_popul[i])
        
        m.exc_nrn_popul = SimObjectPopulation(self.net, exc_nrn_ids)
        
        for i in range(p.nExcNeurons,m.input_nrn_popul.size()):
            inh_nrn_ids.append(m.input_nrn_popul[i])
        
        m.inh_nrn_popul = SimObjectPopulation(self.net, inh_nrn_ids)
                
        return m
    
    
    def scriptList(self):
        return ["PoissInputModel.py"]
    
    
    def getOutput(self, p):
        return self.elements
