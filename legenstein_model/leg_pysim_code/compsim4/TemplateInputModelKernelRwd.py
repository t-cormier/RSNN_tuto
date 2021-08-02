import sys
from pypcsimplus import *
import pypcsimplus as pcsim
from numpy import *
import matplotlib
matplotlib.use('Agg')
sys.path.append('../packages')
from pyV1.inputs import jitteredtemplate as STempl

def rew_kernel(x):
    Apos = 1
    Aneg = -1
    tau_down = 50e-3  
    tau_up = 5e-3 
    if x > 0:
        return Apos * (exp( -x/tau_down ) - exp( - x/tau_up ) ) 
    elif x < 0:
        return Aneg * ( exp( x/tau_down ) - exp( x/tau_up ) )  

class TemplateInputModelKernelRwd(pcsim.Model):
    
    def defaultParameters(self):
        p = self.params
        
        p.nInputChannels = 200
        
        p.templDuration = 500e-3
        p.nTemplates = 2
        p.jitter =  0e-3
        p.templRate = 3
        
        p.numSpikesPerChannel = 1
        
        p.targetTemplate = 0
        
        p.initT = 50e-3
        p.rewardT = 50e-3
        p.rewardDuration = 1000e-3
        
        p.synTauExc = 3e-3
        p.delay = 1e-3
        
        p.Wscale = 0.02
        
        p.WExcScale = 1.0
        p.WInhScale = 1.0
        
        p.connP = 0.2
        
        p.W_Heter = 1.0
        
        p.rewardDelay = 0.3
        p.rewTau = 100e-3
        p.rewPulseScale = 1e-4
        
        p.spikeGeneration = 'fixedSpikesPerChannel'
        
            
        return p
    
    def derivedParameters(self):
        p = self.params
        ep = self.expParams
        dm = self.depModels
        m = self.elements
        net = self.net
        
        p.posRewLevels  = [0,1,0]
        p.posRewDurations = [p.initT, p.rewardT + p.rewardDuration, ep.trialT ] 
        
        p.negRewLevels = [0,-1.0, 0]
        p.negRewDurations = [p.initT, p.rewardT + p.rewardDuration, ep.trialT ]
        
        
    def reset(self, epoch):
        m = self.elements
        p = self.params
        ep = self.expParams
        net = self.net
        
        m.currTemplate = (m.currTemplate+1) % p.nTemplates
        
        stim = m.spiketemplate.generate([m.currTemplate])
        
        
        for ch in stim.channel:
            ch.data = array(ch.data) + p.initT + epoch * ep.trialT
        
        for i in range(m.input_channel_popul.size()):
            if m.input_channel_popul.object(i):                
                m.input_channel_popul.object(i).setSpikes(stim.channel[i].data)
                m.input_channel_popul.object(i).reset(ep.DTsim, epoch * ep.trialT)
    
        
        if m.currTemplate == p.targetTemplate:     
            net.object(m.rewardgen).W = abs(net.object(m.rewardgen).W)
        else:
            net.object(m.rewardgen).W = - abs(net.object(m.rewardgen).W)
        
        if (net.object(m.rewardgen)):            
            net.object(m.rewardgen).reset(ep.DTsim)
            
        m.chosenTemplates.append(m.currTemplate)
    
    
    #
    # Generate the model
    #    
    def generate(self):
        p = self.params
        net = self.net
        m = self.elements
        dm = self.depModels
        
        self.derivedParameters()
        
        # connect the input neurons to the liquid        
        m.input_channel_popul = SimObjectPopulation(net, SpikingInputNeuron(), p.nInputChannels)
        
        m.templ_spikes = []
        
        for tmpl_i in range(p.nTemplates):
            m.templ_spikes.append([ [] for i in range(p.nInputChannels) ])
            for ch_i in range(p.nInputChannels):    
                for n in range(p.numSpikesPerChannel):            
                    m.templ_spikes[tmpl_i][ch_i].append(random.uniform(0, p.templDuration))
        
        for tmpl_i in range(p.nTemplates):
            for ch_i in range(p.nInputChannels):
                m.templ_spikes[tmpl_i][ch_i].sort()
        
        m.spiketemplate = STempl.JitteredTemplate(Tstim=p.templDuration, nChannels=p.nInputChannels, nTemplates=[p.nTemplates], jitter=p.jitter, freq=[p.templRate])
        
        if p.spikeGeneration == "fixedSpikesPerChannel":        
            for tmpl_i in range(p.nTemplates):
                for ch_i in range(p.nInputChannels):                    
                    m.spiketemplate.segment[0].template[tmpl_i].st[ch_i] = m.templ_spikes[tmpl_i][ch_i]
            
        
        m.currTemplate = -1
        
        m.chosenTemplates = [] 
        
        # Create the reward generator                
        m.rewardgen = net.create( StaticCurrAlphaSynapse(1/(p.rewTau*exp(1)) * p.rewPulseScale, tau = p.rewTau, delay = 0), SimEngine.ID(0,0) )
        
        return self.elements
    
    
    def connectReadout(self, readout):
        net = self.net        
        p = self.params
        m = self.elements
        readout_nrn = readout.elements.learning_nrn
        net.connect(m.rewardgen, readout.elements.learning_nrn, Time.sec(p.rewardDelay))
        net.connect(readout.elements.learning_nrn, m.rewardgen, Time.sec(0))
        
     
    def setupRecordings(self):
        m = self.elements
        p = self.params
        ep = self.expParams
        #
        # Recording all the weights
        # 
        r = Recordings(self.net)
        
        r.input_channels = m.input_channel_popul.record(SpikeTimeRecorder())
        
        spikeTemplateWrap = Dictionary()
        
        spikeTemplateWrap.templates = [ [] for i in range(p.nTemplates) ]
        
        for tmpl_i in range(p.nTemplates):
            for ch_i in range(p.nInputChannels):
                spikeTemplateWrap.templates[tmpl_i].append(array(m.spiketemplate.segment[0].template[tmpl_i].st[ch_i]))
        
        r.spikeTemplate = spikeTemplateWrap
            
        r.chosenTemplates = m.chosenTemplates
        
        return r
    
    def scriptList(self):
        return ["TemplateInputModelKernelRwd.py"]

