from pypcsimplus import *
import pypcsimplus as pcsim
from numpy import *
import numpy

class ReadoutModel(pcsim.Model):
    
    def defaultParameters(self):
        p = self.params         
        
        # STDP Parameters
        p.Mu = 0.0013
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
        p.Iinject = 0.0e-10
        
               
        p.Inoise = 0.0e-10
        
        
        
        p.Wscale = 0.184
        p.WExcScale = 1.0
        p.WInhScale = 1.0
        
        p.initLearnWVar = 1.0 / 10.0
        p.initLearnWBound = 2.0 / 10.0
        
        p.initInhWMean = 1.0/ 2.0
        p.initInhWVar = 1.0/10
        p.initInhWBound = 2.0/10
        
        p.diminishingNoise = False
        
        p.noiseSegments = 10
        
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
        
        p.stdpApos = p.Mu * p.Wmax  
        p.stdpAneg = - p.alpha * p.stdpApos
        
        p.samplingTime = int(ep.Tsim / (200 * ep.DTsim))  # sampling time for the histogram in number of simulation steps
        
        
        print "Wmax = ", p.Wmax
    
    
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
        # Setup the neurons
        #*************************************
        self.derivedParameters()
        
        p.NumSyn = dm.input_channel_popul.size()
        
        p.numInhibSynapses = 0 # dm.inh_nrn_popul.size() / 2
        
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
        
        
        if p.DATraceShape == 'alpha':
            DATraceResponse = AlphaFunctionSpikeResponse(p.DATraceTau)
        else:
            DATraceResponse = ExponentialDecaySpikeResponse(p.DATraceTau)
            
        exc_permutation = numpy.random.permutation(dm.input_channel_popul.size())
            
        read_exc_nrns = exc_permutation[:p.NumSyn]        
        
        
        # ******************************** Add learning synapses to learning_nrn
        m.learning_plastic_syn = []        
        for i in xrange(p.NumSyn):
            m.learning_plastic_syn.append(net.connect(dm.input_channel_popul[read_exc_nrns[i]], m.learning_nrn, DAModStdpDynamicCondExpSynapse(
                                                                                          Winit = m.learnSynW[i],
                                                                                          Erev = p.ErevExc, 
                                                                                          U = p.U, 
                                                                                          D = p.D, 
                                                                                          F = p.F, 
                                                                                          Wex = 1.0 * p.Wmax, 
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
        
        if p.diminishingNoise == True:
            m.noise_level_gen = net.add( AnalogLevelBasedInputNeuron(p.noiseLevels, p.noiseDurations), SimEngine.ID(0,0) )
            net.connect(m.noise_level_gen, 0, m.learning_nrn, "Inoise", Time.sec(ep.minDelay))
        
        
        return self.elements

    def setTestPhase(self):
        ep = self.expParams
        net = self.net
        m = self.elements
        if not ep.testWithNoise:
            if (net.object(m.learning_nrn)):
                net.object(m.learning_nrn).Inoise = 0
        for s in m.learning_plastic_syn:
            if (net.object(s)):
                net.object(s).activeDASTDP = False
                
    def setTrainPhase(self):
        net = self.net
        m = self.elements
        ep = self.expParams
        if not ep.testWithNoise:
            if net.object(m.learning_nrn):
                net.object(m.learning_nrn).Inoise = self.params.Inoise
        for s in m.learning_plastic_syn:
            if (net.object(s)):
                net.object(s).activeDASTDP = True
                
    def switchOffRecordVmReadout(self):
        self.net.object(self.recordings.learning_nrn_vm).setActive(False)
        
    def switchOnRecordVmReadout(self):
        self.net.object(self.recordings.learning_nrn_vm).setActive(True)
        
    def increaseThreshold(self):
        self.net.object(self.elements.learning_nrn).Vthresh = 0
        
    def setNormalThreshold(self):
        self.net.object(self.elements.learning_nrn).Vthresh = self.params.Vthresh  
        
    def printSamplingTime(self):
        print "samplingTime is ", self.net.object(self.recordings.learning_nrn_vm).samplingTime
    
     
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
        if ep.recordReadoutVm:
            r.learning_nrn_vm = self.net.record(m.learning_nrn, "Vm", AnalogRecorder())
        else:
            r.learning_nrn_vm = self.net.record(m.learning_nrn, "Vm", AnalogRecorder(p.samplingTime))
        
        return r
    
    def scriptList(self):
        return ["ReadoutModel.py"]