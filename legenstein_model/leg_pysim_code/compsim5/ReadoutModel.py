from pypcsimplus import *
import pypcsimplus as pcsim
import sys
sys.path.append('../packages/reward_gen/build')
from pyreward_gen import *
from numpy import *
import numpy

def rew_kernel(x):
    Apos = 1
    Aneg = -1
    tau_down = 100e-3  
    tau_up = 5e-3 
    if x > 0:
        return Apos * (exp( -x/tau_down ) - exp( - x/tau_up ) ) 
    elif x < 0:
        return Aneg * ( exp( x/tau_down ) - exp( x/tau_up ) )  


class RewardInputModel(pcsim.Model):
    
    def defaultParameters(self):
        p = self.params
        
        p.targetDigit = 2
        p.nDigits = 2
        
        p.rewardT = 0e-3
        p.rewardDuration = 1
        
        p.rewardType = 'pulse'
        p.rewardKernel = 'alpha'
        
        p.nExcNeurons = 100
        p.nInhNeurons = 20
        
        p.rewTau = 100e-3
        p.rewPulseScale = 1e-4
        p.rewardDelay = 0.3
        
        return p
    
    def derivedParameters(self):
        p = self.params
        ep = self.expParams
        if p.rewardType == 'dblexp':        
            rewardKernelLevels = [ rew_kernel(x) for x in arange(-(p.rewardT+ep.initT), 2*p.rewardDuration, ep.DTsim) ]            
            rewardKernelDurations = [ ep.DTsim for i in range(len(dblexpRewardLevels)) ]
        else:
            rewardKernelLevels = [0,1]            
            rewardKernelDurations = [ ep.initT, p.rewardT + p.rewardDuration ]            
            
        p.posRewLevels  = rewardKernelLevels + [0]
        p.posRewDurations = rewardKernelDurations + [ep.trialT]
            
        p.negRewLevels = [0,-1, 0]
        p.negRewDurations = [ep.initT, p.rewardT + p.rewardDuration, ep.trialT ]
        
    def connectReadout(self, readout):
        net = self.net        
        p = self.params
        m = self.elements
        
        net.connect(m.rewardgen, readout.elements.learning_nrn, Time.sec(p.rewardDelay))
        net.connect(readout.elements.learning_nrn, m.rewardgen, Time.sec(0))
        
    
    def generate(self):
        p = self.params
        net = self.net
        m = self.elements
        self.derivedParameters()
        
        m.exc_nrn_popul = SimObjectPopulation(self.net, SpikingInputNeuron(), p.nExcNeurons)
        m.inh_nrn_popul = SimObjectPopulation(self.net, SpikingInputNeuron(), p.nInhNeurons)
        
        # Create the reward generator
        
        if p.rewardKernel == 'alpha':                    
            m.rewardgen = net.create( StaticCurrAlphaSynapse(1/(p.rewTau*exp(1)) * p.rewPulseScale, tau = p.rewTau, delay = 0), SimEngine.ID(0,0) )            
        else:
            m.reward_pulse_gen = net.create( AnalogLevelBasedInputNeuron(p.posRewLevels, p.posRewDurations ), SimEngine.ID(0,0) )
            m.rewardgen = net.create( ReadoutRewardGen(), SimEngine.ID(0,0) )
            net.connect(m.reward_pulse_gen, 0, m.rewardgen, 1, Time.sec(0))
        
        m.currTemplate = -1


    
    def reset(self, trialN, liq_resp, currDigit):
        ep = self.expParams
        m = self.elements
        p = self.params
        net = self.net
                
        for i in range(m.exc_nrn_popul.size()):
            if m.exc_nrn_popul.object(i):                
                m.exc_nrn_popul.object(i).setSpikes(liq_resp.exc_spikes[i] + trialN * ep.trialT + ep.initT)
                m.exc_nrn_popul.object(i).reset(ep.DTsim, trialN * ep.trialT)
                
        for i in range(m.inh_nrn_popul.size()):
            if m.inh_nrn_popul.object(i):                
                m.inh_nrn_popul.object(i).setSpikes(liq_resp.inh_spikes[i] + trialN * ep.trialT + ep.initT)
                m.inh_nrn_popul.object(i).reset(ep.DTsim, trialN * ep.trialT)
          
        if not p.rewardKernel == 'alpha':
            if currDigit == p.targetDigit:     
                net.object(m.reward_pulse_gen).setAnalogValues(p.posRewLevels, p.posRewDurations)
            else:
                net.object(m.reward_pulse_gen).setAnalogValues(p.negRewLevels, p.negRewDurations)
            
            if (net.object(m.reward_pulse_gen)):            
                net.object(m.reward_pulse_gen).reset(ep.DTsim)
        else:                
            if currDigit == p.targetDigit:     
                net.object(m.rewardgen).W = abs(net.object(m.rewardgen).W)
            else:
                net.object(m.rewardgen).W = -abs(net.object(m.rewardgen).W)
            
            if (net.object(m.rewardgen)):            
                net.object(m.rewardgen).reset(ep.DTsim)
        
            
        
    
    def setupRecordings(self):
        m = self.elements
        r = Recordings(self.net)
        ep = self.expParams
        r.exc_spikes = m.exc_nrn_popul.record(SpikeTimeRecorder())
        r.inh_spikes = m.inh_nrn_popul.record(SpikeTimeRecorder())
        return r
    

class ReadoutModel(pcsim.Model):
    
    def defaultParameters(self):
        p = self.params         
        
        # STDP Parameters
        p.Mu = 0.0008
        p.alpha = 1.05
        p.stdpTaupos = 30e-3
        p.stdpTauneg = 30e-3
        p.stdpGap = 5e-4
        
        # Dopamine Modulated STDP Parameters
        p.DATraceDelay = 0.0
        p.DATraceTau = 0.4
        p.DAStdpRate = 3
        p.DATraceShape = 'alpha'
                
        p.KappaAnegSquare = -1.0
        
        # synapse parameters
        p.synTau = 5e-3    
        p.delaySyn = 1e-3
        p.U = 0.5
        p.D = 1.1
        p.F = 0.02
        p.Uinh = 0.25
        p.Dinh = 0.7
        p.Finh = 0.02
        p.ErevExc = 0.0
        p.ErevInh = -75e-3
        
        # Neuron parameters 
        p.Cm = 3e-10
        p.Rm = 1e8 
        p.Vthresh = - 59e-3
        p.Vresting = - 70e-3
        p.Vreset = -70e-3    
        p.Trefract = 5e-3
        p.Iinject = 0.0
        p.Inoise = 0.0
        
        
        p.Wscale = 0.06
        p.WExcScale = 1.0
        p.WInhScale = 1.0
        
        p.initLearnWVar = 1.0 / 10.0
        p.initLearnWBound = 2.0 / 10.0
        
        p.initInhWMean = 1.0/ 2.0
        p.initInhWVar = 1.0/10
        p.initInhWBound = 2.0/10
        
        p.noiseSegments = 10
        
        p.decreaseWeightRate = 0.0 
        
        p.noiseType = 'OU'
        p.OUScale = 0.2
        
        return p
    
    
    def derivedParameters(self):
        p = self.params
        ep = self.expParams
        dm = self.depModels
        m = self.elements
        net = self.net
        
        
    
        p.noiseLevels = [ p.Inoise * (10 - i) / 10 for i in arange(p.noiseSegments) ]
        p.noiseDurations = [ ep.nTrainEpochs * ep.trialT / p.noiseSegments for i in range(p.noiseSegments) ] 
        
        p.synTauInh = 2 * p.synTau
        
        p.Vinit    = p.Vreset
        
        # setup the weights
        tau_m = p.Cm * p.Rm
        tau_s = p.synTau
        p.weightExc = ((p.Vthresh - p.Vinit) * p.WExcScale * p.Wscale)/ ((p.ErevExc - p.Vinit) * p.Rm * tau_s / (tau_m - tau_s) *  ((tau_s / tau_m) ** (tau_s / (tau_m - tau_s)) - (tau_s / tau_m) ** (tau_m / (tau_m - tau_s))))
    
        tau_s = p.synTauInh    
        p.weightInh =  ((p.Vthresh - p.Vinit) * p.WInhScale * p.Wscale)/ (((p.Vinit+ p.Vthresh) / 2 - p.ErevInh) * p.Rm * tau_s / (tau_m - tau_s) *  ((tau_s / tau_m) ** (tau_s / (tau_m - tau_s)) - (tau_s / tau_m) ** (tau_m / (tau_m - tau_s))))
        
        p.Wmax = p.weightExc * 2.5
        p.WmaxInh = p.weightInh * 2.5
        
        p.stdpApos = p.Mu * p.Wmax # is actually multiplication of the learning rate mu and Apos from stdp 
        p.stdpAneg = - p.alpha * p.stdpApos
        
        p.samplingTime = int(ep.Tsim / (200 * ep.DTsim))  # sampling time for the histogram in number of simulation steps
        
        
        print "Wmax = ", p.Wmax
        
    def diminishWeights(self):
        net = self.net
        m = self.elements
        p = self.params
        for id in m.learning_plastic_syn:
            net.object(id).W -= net.object(id).W * p.decreaseWeightRate
        
    
    #
    # Generate the model
    #    
    def generate(self):
        p = self.params
        ep = self.expParams
        dm = self.depModels
        m = self.elements
        net = self.net
        #*************************************
        # Setup the readout neuron
        #*************************************
        self.derivedParameters()
        
        p.NumSyn = dm.exc_nrn_popul.size()
        
        p.numInhibSynapses = 0 
        
        m.learnSynW = random.normal(1.0/2 * p.Wmax, p.initLearnWVar * p.Wmax, p.NumSyn)        
        m.learnSynW.clip( min = (1.0/2 - p.initLearnWBound )* p.Wmax , max = (1.0/2 + p.initLearnWBound )* p.Wmax)
        
        m.inhibSynW = random.normal(p.initInhWMean * p.WmaxInh, p.initInhWVar * p.WmaxInh,p.numInhibSynapses) 
        m.inhibSynW.clip( min = (p.initInhWMean + p.initInhWBound) * p.WmaxInh, max = (p.initInhWMean - p.initInhWBound) * p.WmaxInh)
        
        
        m.learning_nrn = net.add( DARecvCbLifNeuron(Cm = p.Cm, 
                                                 Rm = p.Rm, 
                                                 Vresting = p.Vresting, 
                                                 Vthresh  = p.Vthresh, 
                                                 Vreset   = p.Vreset, 
                                                 Vinit    = p.Vinit, 
                                                 Trefract = p.Trefract, 
                                                 Iinject = p.Iinject, 
                                                 Inoise = p.Inoise), SimEngine.ID(0, 0) )
        
        if p.noiseType == 'OU':
            net.mount(OUNoiseSynapse(0.012e-6 * p.OUScale, 0.003e-6 * p.OUScale, 2.7e-3, 0.0), m.learning_nrn)
            net.mount(OUNoiseSynapse(0.057e-6 * p.OUScale, 0.0066e-6 * p.OUScale, 10.5e-3,-75e-3), m.learning_nrn)
        
        
        # Connect the learning neurons to the liqduid
        if p.DATraceShape == 'alpha':
            DATraceResponse = AlphaFunctionSpikeResponse(p.DATraceTau)
        else:
            DATraceResponse = ExponentialDecaySpikeResponse(p.DATraceTau)
            
        exc_permutation = numpy.random.permutation(dm.exc_nrn_popul.size())
            
        read_exc_nrns = exc_permutation[:p.NumSyn]        
        read_inh_nrns = numpy.random.permutation(dm.inh_nrn_popul.size())[:p.numInhibSynapses]
        
        # ******************************** Add learning synapses to learning_nrn
        m.learning_plastic_syn = []        
        for i in xrange(p.NumSyn):
            m.learning_plastic_syn.append(net.connect(dm.exc_nrn_popul[read_exc_nrns[i]], m.learning_nrn, DAModStdpDynamicCondExpSynapse(
                                                                                          Winit = m.learnSynW[i],
                                                                                          tau = p.synTau,
                                                                                          delay = p.delaySyn,
                                                                                          Erev = p.ErevExc, 
                                                                                          U = p.U, 
                                                                                          D = p.D, 
                                                                                          F = p.F, 
                                                                                          Wex = p.Wmax, 
                                                                                          activeDASTDP = True, 
                                                                                          STDPgap = p.stdpGap, 
                                                                                          Apos = p.stdpApos, 
                                                                                          Aneg = p.stdpAneg, 
                                                                                          taupos = p.stdpTaupos, 
                                                                                          tauneg = p.stdpTauneg, 
                                                                                          DATraceDelay = p.DATraceDelay, 
                                                                                          DAStdpRate = p.DAStdpRate, 
                                                                                          useFroemkeDanSTDP = False, 
                                                                                          daTraceResponse = DATraceResponse)))
                
        m.inhib_learn_syn = []
        for i in xrange(p.numInhibSynapses):
            m.inhib_learn_syn.append(net.connect(dm.inh_nrn_popul[read_inh_nrns[i]], m.learning_nrn, DynamicCondExpSynapse(W = inhibSynW[i], 
                                                                                                 delay = p.delaySyn, 
                                                                                                 tau = p.synTauInh, 
                                                                                                 Erev = p.ErevInh, 
                                                                                                 U = p.Uinh, 
                                                                                                 D = p.Dinh, 
                                                                                                 F = p.Finh)))
        
        return self.elements
    
    
    def switchOffRecordVmReadout(self):
        self.net.object(self.recordings.learning_nrn_vm).setActive(False)
        
    def switchOnRecordVmReadout(self):
        self.net.object(self.recordings.learning_nrn_vm).setActive(True)
        
    def increaseThreshold(self):
        self.net.object(self.elements.learning_nrn).Vthresh = 0
        
    def setNormalThreshold(self):
        self.net.object(self.elements.learning_nrn).Vthresh = self.params.Vthresh  
    
    
    def setTestPhase(self):
        net = self.net
        m = self.elements        
        for s in m.learning_plastic_syn:
            if (net.object(s)):
                net.object(s).activeDASTDP = False
    
    
    def deactivateLearning(self):
        net = self.net
        m = self.elements        
        for s in m.learning_plastic_syn:
            if (net.object(s)):
                net.object(s).activeDASTDP = False
    
                
                
    def setTrainPhase(self):
        net = self.net
        m = self.elements        
        for s in m.learning_plastic_syn:
            if (net.object(s)):
                net.object(s).activeDASTDP = True
                
    def activateLearning(self):
        net = self.net
        m = self.elements        
        for s in m.learning_plastic_syn:
            if (net.object(s)):
                net.object(s).activeDASTDP = True
    
     
    def setupRecordings(self):
        m = self.elements
        p = self.params
        ep = self.expParams
        #
        # Recording all the weights
        # 
        r = Recordings(self.net)
        self.recordings = r
         
        r.weights = SimObjectPopulation(self.net, m.learning_plastic_syn).record(AnalogRecorder(p.samplingTime), "W")         
        
        r.learning_spikes =  self.net.record(m.learning_nrn, SpikeTimeRecorder())
        if ep.runMode.startswith("short"):
            r.learning_nrn_vm = self.net.record(m.learning_nrn, "Vm", AnalogRecorder())
        else:
            r.learning_nrn_vm = self.net.record(m.learning_nrn, "Vm", AnalogRecorder(p.samplingTime))
        
        return r
    
    def scriptList(self):
        return ["ReadoutModel.py"]