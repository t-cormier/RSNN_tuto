#======================================================================
#  Computer Simulation 3 from
#      Legenstein, R., Pecevski, D. and Maass, W., A Learning Theory  
#      for Reward-Modulated Spike-Timing-Dependent Plasticity with 
#      Applicastion to Biofeedback
# 
#  Author: Dejan Pecevski, dejan@igi.tugraz.at
#
#  Date: March 2008
#
#======================================================================

import sys
import os

from pypcsim import *
sys.path.append('../packages/reward_gen/build')
from pyreward_gen import *
from numpy import *
import random, getopt
from datetime import datetime
from math import *
from tables import *
import Constraints

def experiment(params = {}, expname = 'noname'):
    
    class Parameters:
        pass
    
    p = Parameters()
    
    random.seed( datetime.today().microsecond )
    p.pyRandomSeed = 534983
    random.seed( p.pyRandomSeed )
    tstart=datetime.today()
    
    p.Tsim = 20*3600
    p.DTsim = 1e-4    
    p.NumSyn = 100
    p.ratioStrong = 0.5
    p.numAdditionalTargetSynapses = 10
    
    # synapse parameters
    p.synTau = 7e-3
    p.delaySyn = 1e-3
    p.wScale = 2.0
    p.inputRate = 6
    
    p.constructionSeed = 3273435
    p.simulationSeed = 132929
    
    # neuron parameters
    p.C = 1
    p.Rm = 1
    p.Trefract = 0.2e-3
    p.Rbase = p.Iinject = 5
    
    # STDP parameters
    p.stdpA = 0.0002 * 2.77    
    p.alpha = 1.07
    p.stdpTaupos = 25e-3
    p.stdpTauneg = 25e-3
    
    # Da modulated STDP parameters
    p.DATraceDelay = 0.0
    p.DATraceTau = 0.4
    p.DAStdpRate = 1.0
    p.DATraceShape = 'alpha'
    
    # reward kernel parameters
    p.rewardDelay = 0.4 # In seconds
    p.KappaAlpha = 1.12 
    p.KappaTaupos = 20e-3
    p.KappaTauneg = 20e-3
    p.KappaTaupos2 = 4e-3    
    p.KappaTauneg2 = 4e-3
    p.KappaTauposSquare = 50e-3
    p.KappaTaunegSquare = 50e-3
    
    
    p.directory = '.'
    
    # override the source parameters    
    for pname,value in params.iteritems():
        print "new value ", pname, value
        if not hasattr(p,pname):
            raise AttributeError("Parameter "+ pname + " not existent!")
        setattr(p,pname,value)
    
    
    
    p.minDelay = p.DTsim
    p.numStrongTargetSynapses = int( p.NumSyn * p.ratioStrong )
    p.numWeakTargetSynapses = p.NumSyn - p.numStrongTargetSynapses
    
    p.WmaxTrue = p.wScale/p.NumSyn
    p.Wmax = p.WmaxTrue/p.synTau
    
    learnSynW = [ random.gauss(1.0/2 * p.Wmax, 1.0/10 * p.Wmax) for i in range(p.NumSyn) ]
    learnSynW = [ (lambda x : min(7.0/10 * p.Wmax, max(x, 3.0/10 * p.Wmax)))(x) for x in learnSynW ]
    targetSynW = [ p.Wmax for i in range(p.numStrongTargetSynapses) ]
    
    additionalTargetSynW = [ p.Wmax for i in range(p.numAdditionalTargetSynapses) ]
    
    # STDP parameters
    p.stdpApos = p.stdpA * p.Wmax
    p.stdpApos = p.stdpA * p.Wmax    
    p.stdpAneg = - p.alpha * p.stdpApos
    p.stdpGap = p.DTsim
        
    p.stdpAposTrue = p.stdpApos * p.synTau
    p.stdpAnegTrue = p.stdpAneg * p.synTau
        
    p.kappaScalePos = p.KappaTauposSquare / (p.KappaTaupos - p.KappaTaupos2)
    p.KappaApos = p.KappaAlpha * p.kappaScalePos
        
    p.kappaScaleNeg = p.KappaTaunegSquare / (p.KappaTauneg - p.KappaTauneg2)                  
    p.KappaAneg = - 1.0 * p.kappaScaleNeg
    
    p.KernelType = 'DblExp'
    p.KappaGap = p.DTsim
    
    p.burstT = -1
    p.burstEventRate = -1
    p.burstHighRate = -1
    p.burstLowRate = -1
    
    p.KappaTe = Constraints.optimal_Te_value(0.1, 0.0001, p.KappaApos, p.KappaAneg, p.KappaTaupos, p.KappaTauneg, p.KappaTaupos2, p.KappaTauneg2, p.KernelType, p.synTau)
    
    p.samplingTime = int(p.Tsim / (200 * p.DTsim))  # sampling time for the histogram in number of simulation steps
    
    sp = SimParameter( dt=Time.sec( p.DTsim ) , minDelay = Time.sec(p.minDelay), maxDelay = Time.sec(2), simulationRNGSeed = p.simulationSeed, constructionRNGSeed = p.constructionSeed );
    net = SingleThreadNetwork( sp )
    
    
    input_nrn_popul = SimObjectPopulation( net, PoissonInputNeuron(p.inputRate, p.Tsim, 0 ), p.NumSyn )
        
    additional_input_nrn_popul = SimObjectPopulation(  net, PoissonInputNeuron(p.inputRate, p.Tsim,0), p.numAdditionalTargetSynapses )    

    
    learning_nrn = net.add( DARecvLinearPoissonNeuron(p.C, p.Rm, p.Trefract, 0, p.Iinject) )
    target_nrn = net.add( DARecvLinearPoissonNeuron(p.C, p.Rm, p.Trefract, 0, 0) )
    
    DATraceResponse = AlphaFunctionSpikeResponse( p.DATraceTau )
    
    learning_plastic_syn = []        
    for i in xrange(p.NumSyn):
        learning_plastic_syn.append( net.connect(input_nrn_popul[i], learning_nrn, DAModulatedStaticStdpSynapse(Winit = learnSynW[i],
                                                                                   tau = p.synTau,
                                                                                   delay = p.delaySyn,
                                                                                   Wex = p.Wmax, 
                                                                                   STDPgap = p.stdpGap,
                                                                                   Apos = p.stdpApos,
                                                                                   Aneg = p.stdpAneg, 
                                                                                   taupos = p.stdpTaupos,
                                                                                   tauneg = p.stdpTauneg,
                                                                                   DATraceDelay = p.DATraceDelay,
                                                                                   DAStdpRate = p.DAStdpRate, 
                                                                                   useFroemkeDanSTDP = False,
                                                                                   daTraceResponse = DATraceResponse ) ) )
        
        
    
    target_syn = []
    for i in xrange(p.numStrongTargetSynapses):
        target_syn.append( net.connect(input_nrn_popul[i], target_nrn, StaticSpikingSynapse(W = targetSynW[i], 
                                                                                         delay = p.delaySyn,
                                                                                         tau = p.synTau) ) )
        
    addit_target_syn = []
    for i in xrange(p.numAdditionalTargetSynapses):
        addit_target_syn.append(net.connect(additional_input_nrn_popul[i], target_nrn,
                                                   StaticSpikingSynapse(W = additionalTargetSynW[i], 
                                                                        delay = p.delaySyn,
                                                                        tau = p.synTau) ) )
    
    
    rewardGenFactory = BioFeedRewardGenDblExp( Apos = p.KappaApos,
                                                Aneg = p.KappaAneg,
                                                taupos1 = p.KappaTaupos,
                                                tauneg1 = p.KappaTauneg,
                                                taupos2 = p.KappaTaupos2,
                                                tauneg2 = p.KappaTauneg2,
                                                Gap = p.KappaGap,                                             
                                                Te = p.KappaTe )      
    
    
    # Create the reward generator and connect it in the circuit
    reward_gen = net.add( rewardGenFactory )
    
    net.connect(learning_nrn, 0, reward_gen, 1, Time.sec(p.minDelay))
    net.connect(target_nrn, 0, reward_gen, 0, Time.sec(p.minDelay))    
    net.connect(reward_gen, 0, learning_nrn, 0, Time.sec(p.rewardDelay))
        
    #
    # Recording all the weights
    #    
    synWeightRec = net.create( AnalogRecorder(p.samplingTime), p.NumSyn )
    
    for i in range(p.NumSyn):
        net.connect(learning_plastic_syn[i], "W", synWeightRec[i], 0, Time.sec(0))
    
    # Recorders for the two neurons
    target_nrn_rec = net.add( SpikeTimeRecorder() )
    learning_nrn_rec = net.add( SpikeTimeRecorder() )
    input_nrn_rec = net.add( SpikeTimeRecorder() )
    
    net.connect( target_nrn, target_nrn_rec, Time.sec(p.minDelay) )
    net.connect( learning_nrn, learning_nrn_rec, Time.sec(p.minDelay) )
    net.connect( input_nrn_popul[0], input_nrn_rec, Time.sec(p.minDelay) )
    
    
    # Run simulation 
    print 'Running simulation:';
    t0=datetime.today()
    
    net.add( SimProgressBar( Time.sec(p.Tsim) ) )
    
    print "Simulation start : " , datetime.today().strftime('%x %X')
    net.reset();
    net.advance( int( p.Tsim / p.DTsim ) )
    
    t1=datetime.today()
    print 'Done.', (t1-t0).seconds, 'sec CPU time for', p.Tsim*1000, 'ms simulation time';
    print '==> ', (t1-tstart).seconds, 'seconds total'
    
    
    weights = vstack([ array(net.object(synWeightRec[i]).getRecordedValues()) for i in range(p.NumSyn) ])
    
    strong_syn_avg = mean(weights[:p.numStrongTargetSynapses], 0)
    strong_syn_std = std(weights[:p.numStrongTargetSynapses], 0)
    weak_syn_avg =   mean(weights[p.numStrongTargetSynapses:p.NumSyn], 0)
    weak_syn_std =   std(weights[p.numStrongTargetSynapses:p.NumSyn], 0)

    learning_nrn_spikes = array( net.object(learning_nrn_rec).getSpikeTimes() )
    target_nrn_spikes = array( net.object(target_nrn_rec).getSpikeTimes() )
    input_nrn_spikes = array( net.object(input_nrn_rec).getSpikeTimes() )
    
    learningNrnRate = float(len(learning_nrn_spikes))/ p.Tsim
    targetNrnRate = float(len(target_nrn_spikes)) / p.Tsim
    
    # Save results
    print "Saving results..."
    hostname = os.environ['HOSTNAME'].split('.')[0]        
    h5file = openFile( os.path.join(p.directory,"biof_poiss_" + expname + "_" + datetime.today().strftime("%Y%m%d_%H%M%S") + '.' + hostname + ".res.h5"), mode = "w", title = "Biofeedback DASTDP Experiment results")
    
    
    cmd = "class ExperimentParameters(IsDescription):\n"    
    for pname in dir(p):
        if not pname.startswith("__"):            
            data_type = type(getattr(p,pname))
            if data_type == type(3):
                cmd_type = 'IntCol()\n'
            elif data_type == type('t'):
                cmd_type = 'StringCol(50)\n'
            else:
                cmd_type = 'FloatCol()\n'
            
            cmd += "\t" + pname + " = " + cmd_type
    
    exec(cmd) in globals()
    paramTable = h5file.createTable("/", "parameters", ExperimentParameters, "Experiment parameters")    
    params = paramTable.row
    # General parameters
    for pname in dir(p):
        if not pname.startswith("__"):
            params[pname] = getattr(p,pname)
    
    params.append()
    paramTable.flush()
    
    avg_syn_group = h5file.createGroup("/", "AverageSynapseWeights", "")
    h5file.createArray(avg_syn_group, "strong_avg",  strong_syn_avg, "")
    h5file.createArray(avg_syn_group, "weak_avg",  weak_syn_avg, "")
    h5file.createArray(avg_syn_group, "strong_std",  strong_syn_std, "")
    h5file.createArray(avg_syn_group, "weak_std",  weak_syn_std, "")
    
    observed_syn_group = h5file.createGroup("/", 'ObservedSynapsesWeights', 'Weights of synapses that are observed')
    wmaxSynRecValues = weights[0:3]
    h5file.createArray(observed_syn_group, "ObservedStrongSynapsesWeights", wmaxSynRecValues, "")
    zeroSynRecValues = weights[p.numStrongTargetSynapses:p.numStrongTargetSynapses+3]
    h5file.createArray(observed_syn_group, "ObservedWeakSynapsesWeights", zeroSynRecValues, "")
    h5file.createArray(observed_syn_group, "weights", weights, "")
    
    spikes_group = h5file.createGroup("/", 'spikes', "Recorded spikes from the learning, target and one input neuron")
    h5file.createArray(spikes_group, "learningNeuronSpikes", learning_nrn_spikes, "")
    h5file.createArray(spikes_group, "targetNeuronSpikes", target_nrn_spikes, "")
    h5file.createArray(spikes_group, "inputNeuronSpikes", input_nrn_spikes, "")
                     
    def save_script_file(h5file,group, scr_filename):
        f = open(scr_filename, "rt")
        fcontent = f.read()
        f.close()
        h5_obj_name = os.path.basename(scr_filename)
        h5_obj_name = h5_obj_name.replace(".", "_")
        h5file.createArray(group, h5_obj_name, fcontent)
        pass
    
    scripts_group = h5file.createGroup("/", "scripts", "")
    save_script_file(h5file, scripts_group, sys.argv[0])
    save_script_file(h5file, scripts_group, "Constraints.py" )
    save_script_file(h5file, scripts_group, "biofeedback_poiss_dbl_exp.py" )
    
    h5file.close();
    print "Done."

if __name__ == '__main__':
    experiment()