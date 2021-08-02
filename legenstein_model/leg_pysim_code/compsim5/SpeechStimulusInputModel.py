import matplotlib
matplotlib.use('Agg')

import pypcsimplus as pcsim
from pypcsimplus import *
import sys
sys.path.append('../packages')
from pyV1.dataformats import *
sys.path.append('../packages/speech')
from speechInput import *   
import time
from StimulusModel import *
from numpy import *

class SpeechStimulusInputModel(StimulusModel):
    
    def defaultParameters(self):
        StimulusModel.defaultParameters(self)
        p = self.params
        
        p.nInputChannels = 10
        
        p.nDigits = 2
        
        p.templDuration = 600e-3
        
        p.initT = 100e-3
        
        p.synTauExc = 3e-3
        p.delay = 1e-3
        
        p.Wscale = 0.06
        p.WExcScale = 1.0
        p.WInhScale = 1.4
        
        p.connP = 0.05        
        p.W_Heter = 0.0
        
        p.speechInputFile = "spkdata_40.h5"
        p.speechFileDir = '.'
            
        return p
    
    def derivedParameters(self):
        p = self.params
        ep = self.expParams
        dm = self.depModels
        m = self.elements
        net = self.net
        
        p.synTauInh = 2 * p.synTauExc
        
        # setup the weights
        tau_m = dm.params.Cm * dm.params.Rm
        tau_s = p.synTauExc
        p.Wexc = ((dm.params.Vthresh - dm.params.Vinit) * p.WExcScale * p.Wscale)/ ((dm.params.ErevExc - dm.params.Vinit) * dm.params.Rm * tau_s / (tau_m - tau_s) *  ((tau_s / tau_m) ** (tau_s / (tau_m - tau_s)) - (tau_s / tau_m) ** (tau_m / (tau_m - tau_s))))
    
        tau_s = p.synTauInh    
        p.Winh =  ((dm.params.Vthresh - dm.params.Vinit) * p.WInhScale * p.Wscale)/ (((dm.params.Vinit + dm.params.Vthresh) / 2 - dm.params.ErevInh) * dm.params.Rm * tau_s / (tau_m - tau_s) *  ((tau_s / tau_m) ** (tau_s / (tau_m - tau_s)) - (tau_s / tau_m) ** (tau_m / (tau_m - tau_s))))
    
    def getStimulusInput(self, SUD):
        return [ self.stimulusArray[SUD].channel[i].data for i in range(len(self.stimulusArray[SUD].channel)) ]         
    
    
    def loadSpeechStimuli(self):
        p = self.params
        rnd_state = random.get_state()
        random.seed()
        time.sleep(random.uniform(0,2))
        random.set_state(rnd_state)
        p.speakersUsed = [1,2,5,6,7]
        
        speechGenerator = PreprocessedSpeech(h5filename = p.speechInputFile, path = p.speechFileDir)
        
        self.stimulusArray = {} 
        for speaker in p.speakersUsed:
            for utter in range(1,11):
                for digit in range(1,3):
                    self.stimulusArray[(speaker,utter,digit)] = speechGenerator.generateBySUD(speaker,utter,digit,False)
                    

        
    def resetStimulus(self, SUD):
        self.stimulus = self.stimulusArray[SUD]
        StimulusModel.generate(self)
        StimulusModel.reset(self)
        
    def generate(self):
        p = self.params
        net = self.net
        m = self.elements
        self.derivedParameters()
                
        dm = self.depModels.elements        
        StimulusModel.generate(self)
        ch_i = 0
        m.syn = []
        coor_z_ranges = [ [0,3] , [3,6] , [0,3], [3,6] ]
        coor_y_ranges = [ [0,3] , [0,3] , [3,6], [3,6] ]
        for i in range(20):                         
            for z in range(coor_z_ranges[i%4][0],coor_z_ranges[i%4][1]):
                for y in range(coor_y_ranges[i%4][0], coor_y_ranges[i%4][1]):
                    for x in  range(3*int(float(i)/4),3*int(float(i)/4)+3):
                        m.syn.append( net.connect(m.popul(4*i+2), dm.all_nrn_popul.getIdAt(x,y,z), StaticSpikingSynapse(W = p.Wexc, tau = p.synTauExc, delay = p.delay )) )
        
        self.loadSpeechStimuli()
         
        
    def generate2(self):        
        p = self.params
        net = self.net
        m = self.elements
        self.derivedParameters()
        
        StimulusModel.generate(self)
        dm = self.depModels.elements
        excFactory = SimObjectVariationFactory( StaticSpikingSynapse(W = p.Wexc, tau = p.synTauExc, delay = p.delay ) )
        
        excFactory.set("W", BndGammaDistribution( p.Wexc, p.W_Heter, 2 * p.Wexc ) )
        
        inhFactory = SimObjectVariationFactory( StaticSpikingSynapse(W = p.Wexc, tau = p.synTauExc, delay = p.delay) )
        
        inhFactory.set("W", BndGammaDistribution( p.Wexc, p.W_Heter, 2 * p.Wexc ) )
        
        m.exc_proj = ConnectionsProjection(m.popul, dm.exc_nrn_popul,
                                             excFactory,
                                             RandomConnections( p.connP ) )
        
        m.inh_proj = ConnectionsProjection(m.popul, dm.inh_nrn_popul,
                                             inhFactory,                                             
                                             RandomConnections( p.connP ) )
        self.loadSpeechStimuli()
        
        
        
    def setupRecordings(self):
        net = self.net
        m = self.elements 
        r = Recordings(net)
        r.input = m.popul.record( SpikeTimeRecorder() )        
        return r
        
    
    def scriptList(self):
        return ["SpeechStimulusInputModel.py"] + StimulusModel.scriptList(self)

    