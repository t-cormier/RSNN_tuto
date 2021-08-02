from pypcsimplus import *
import pypcsimplus as pcsim
from numpy import *
import numpy
from tables import *

class BeforeAfterBiofeedModel(pcsim.Model):
    
    def defaultParameters(self):
         p = self.params         
         p.sampleIdx = 0         
         p.h5filename = last_file('biofeed.*\.h5$')
         return p
    
    #
    # Generate the model
    #    
    def generate(self):
        my_p = self.params
        ep = self.expParams
        dm = self.depModels
        m = self.elements
        net = self.net
        
        print "Opening file ", my_p.h5filename
        p = constructParametersFromH5File(my_p.h5filename).biofeed
        r = constructRecordingsFromH5File(my_p.h5filename).biofeed
        
        if not hasattr(p, "additScale"):
            p.additScale = 1.0
        
        last_weights = []
        initial_weights = []        
        for w in r.weights:
            last_weights.append(mean(w[-1]))
            initial_weights.append(w[my_p.sampleIdx])
            
        beforeLearnSynW = initial_weights
        afterLearnSynW = last_weights        
        
        targetSynW = hstack((ones(p.numStrongTargetSynapses) * p.Wmax, zeros(p.numWeakTargetSynapses)))
        
        inhibSynW = random.normal(p.initInhWMean * p.WmaxInh, p.initInhWVar * p.WmaxInh,p.numInhibSynapses) 
        inhibSynW.clip( min = (p.initInhWMean + p.initInhWBound) * p.WmaxInh, max = (p.initInhWMean - p.initInhWBound) * p.WmaxInh)
        
        additionalTargetSynW = ones(p.numAdditionalTargetSynapses) * p.additScale * p.Wmax
        
        print "Wmax = ", p.Wmax
        
        #*************************************
        # Setup the neurons
        #*************************************
        
        m.before_learning_nrn = net.add(DARecvLifNeuron(Cm = p.Cm, 
                                                 Rm = p.Rm, 
                                                 Vresting = p.Vresting, 
                                                 Vthresh  = p.Vthresh, 
                                                 Vreset   = p.Vreset, 
                                                 Vinit    = p.Vinit, 
                                                 Trefract = p.Trefract, 
                                                 Iinject = 0, 
                                                 Inoise = 0), SimEngine.ID(0, 0 % (net.maxLocalEngineID() + 1)) )

        m.after_learning_nrn = net.add(DARecvLifNeuron(Cm = p.Cm, 
                                                 Rm = p.Rm, 
                                                 Vresting = p.Vresting, 
                                                 Vthresh  = p.Vthresh, 
                                                 Vreset   = p.Vreset, 
                                                 Vinit    = p.Vinit, 
                                                 Trefract = p.Trefract, 
                                                 Iinject = 0, 
                                                 Inoise = 0), SimEngine.ID(0, 0 % (net.maxLocalEngineID() + 1)) )

        
        m.target_nrn = net.add(LifNeuron(Cm = p.Cm, 
                                         Rm = p.Rm, 
                                         Vresting = p.Vresting, 
                                         Vthresh  = p.Vthresh, 
                                         Vreset   = p.Vreset, 
                                         Vinit    = p.Vinit, 
                                         Trefract = p.Trefract), SimEngine.ID(0, 2 % (net.maxLocalEngineID() + 1)))
        
        m.realiz_target_nrn = net.add(LifNeuron(Cm = p.Cm, 
                                         Rm = p.Rm, 
                                         Vresting = p.Vresting, 
                                         Vthresh  = p.Vthresh, 
                                         Vreset   = p.Vreset, 
                                         Vinit    = p.Vinit, 
                                         Trefract = p.Trefract), SimEngine.ID(0, 2 % (net.maxLocalEngineID() + 1)))
        
        
        # Connect the learning and target neurons to the circuit
        if p.DATraceShape == 'alpha':
            DATraceResponse = AlphaFunctionSpikeResponse(p.DATraceTau)
        else:
            DATraceResponse = ExponentialDecaySpikeResponse(p.DATraceTau)
            
        exc_permutation = numpy.random.permutation(dm.exc_nrn_popul.size())
            
        read_exc_nrns = exc_permutation[:p.NumSyn]
        
        addit_read_exc_nrns = exc_permutation[p.NumSyn:(p.NumSyn + p.numAdditionalTargetSynapses)]
        
        read_inh_nrns = numpy.random.permutation(dm.inh_nrn_popul.size())[:p.numInhibSynapses]
        
        # ******************************** Add learning synapses to learning_nrn
        m.before_learning_plastic_syn = []        
        for i in xrange(p.NumSyn):
            m.before_learning_plastic_syn.append(net.connect(dm.exc_nrn_popul[read_exc_nrns[i]], m.before_learning_nrn, DAModulatedStaticStdpSynapse(
                                                                                          Winit= beforeLearnSynW[i], 
                                                                                          tau = p.synTau, 
                                                                                          delay = p.delaySyn, 
                                                                                          Wex = p.Wmax, 
                                                                                          activeDASTDP = False, 
                                                                                          STDPgap = p.stdpGap, 
                                                                                          Apos = p.stdpApos, 
                                                                                          Aneg = p.stdpAneg, 
                                                                                          taupos = p.stdpTaupos, 
                                                                                          tauneg = p.stdpTauneg, 
                                                                                          DATraceDelay = p.DATraceDelay, 
                                                                                          DAStdpRate = p.DAStdpRate, 
                                                                                          useFroemkeDanSTDP = False, 
                                                                                          daTraceResponse = DATraceResponse)))
        
        m.after_learning_plastic_syn = []        
        for i in xrange(p.NumSyn):
            m.after_learning_plastic_syn.append(net.connect(dm.exc_nrn_popul[read_exc_nrns[i]], m.after_learning_nrn, DAModulatedStaticStdpSynapse(
                                                                                          Winit= afterLearnSynW[i], 
                                                                                          tau = p.synTau, 
                                                                                          delay = p.delaySyn,
                                                                                          Wex = p.Wmax, 
                                                                                          activeDASTDP = False, 
                                                                                          STDPgap = p.stdpGap, 
                                                                                          Apos = p.stdpApos, 
                                                                                          Aneg = p.stdpAneg, 
                                                                                          taupos = p.stdpTaupos, 
                                                                                          tauneg = p.stdpTauneg, 
                                                                                          DATraceDelay = p.DATraceDelay, 
                                                                                          DAStdpRate = p.DAStdpRate, 
                                                                                          useFroemkeDanSTDP = False, 
                                                                                          daTraceResponse = DATraceResponse)))
            
                                                                                          
        
        m.target_syn = []
        for i in xrange(p.NumSyn):
            m.target_syn.append(net.connect(dm.exc_nrn_popul[read_exc_nrns[i]], m.target_nrn, StaticSpikingSynapse(W = targetSynW[i], 
                                                                                                 delay = p.delaySyn, 
                                                                                                 tau = p.synTau)))
        
        
        m.realiz_target_syn = []
        for i in xrange(p.NumSyn):
            m.realiz_target_syn.append(net.connect(dm.exc_nrn_popul[read_exc_nrns[i]], m.realiz_target_nrn, StaticSpikingSynapse(W = targetSynW[i], 
                                                                                                 delay = p.delaySyn, 
                                                                                                 tau = p.synTau)))
            
        
            
        m.addit_target_syn = []
        for i in xrange(p.numAdditionalTargetSynapses):
            m.addit_target_syn.append(net.connect(dm.exc_nrn_popul[addit_read_exc_nrns[i]], m.target_nrn, StaticSpikingSynapse(W = additionalTargetSynW[i], 
                                                                                                 delay = p.delaySyn, 
                                                                                                 tau = p.synTau)))
        
                
        m.inhib_before_learn_syn = []
        for i in xrange(p.numInhibSynapses):
            m.inhib_before_learn_syn.append(net.connect(dm.inh_nrn_popul[read_inh_nrns[i]], m.before_learning_nrn, StaticSpikingSynapse(W = inhibSynW[i], 
                                                                                                 delay = p.delaySyn, 
                                                                                                 tau = p.synTauInh)))
        
        m.inhib_after_learn_syn = []
        for i in xrange(p.numInhibSynapses):
            m.inhib_after_learn_syn.append(net.connect(dm.inh_nrn_popul[read_inh_nrns[i]], m.after_learning_nrn, StaticSpikingSynapse(W = inhibSynW[i], 
                                                                                                 delay = p.delaySyn, 
                                                                                                 tau = p.synTauInh)))
            
        m.inhib_target_syn = []
        for i in xrange(p.numInhibSynapses):
            m.inhib_target_syn.append(net.connect(dm.inh_nrn_popul[read_inh_nrns[i]], m.target_nrn, StaticSpikingSynapse(W = inhibSynW[i], 
                                                                                                 delay = p.delaySyn, 
                                                                                                 tau = p.synTauInh)))

        m.inhib_realiz_target_syn = []
        for i in xrange(p.numInhibSynapses):
            m.inhib_realiz_target_syn.append(net.connect(dm.inh_nrn_popul[read_inh_nrns[i]], m.realiz_target_nrn, StaticSpikingSynapse(W = inhibSynW[i], 
                                                                                                 delay = p.delaySyn, 
                                                                                                 tau = p.synTauInh)))
        
    #    
    # Setup the default Recordings for this model (the method can be overriden for inherited classes)
    # this should be invoked from Experiment.setupRecordings
    # 
    def setupRecordings(self):
        m = self.elements
        p = self.params
        ep = self.expParams
        #
        # Recording all the weights
        # 
        r = Recordings(self.net)
        
        # Recorders for the two neurons
        r.target_nrn_spikes = self.net.record(m.target_nrn, SpikeTimeRecorder())
        r.realiz_target_nrn_spikes = self.net.record(m.realiz_target_nrn, SpikeTimeRecorder())
        
        r.before_learning_nrn_spikes =  self.net.record(m.before_learning_nrn, SpikeTimeRecorder())
        r.after_learning_nrn_spikes =  self.net.record(m.after_learning_nrn, SpikeTimeRecorder())
        
        return r
    
    def scriptList(self):
        return ["BeforeAfterBiofeedModel.py"]