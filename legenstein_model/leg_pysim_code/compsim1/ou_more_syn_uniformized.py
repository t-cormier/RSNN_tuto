#================================================================================
#  Computer simulation 1 of 
#      Legenstein, R., Pecevski, D. and Maass, W., A Learning Theory of 
#      Reward-Modulated Spike-Timing-Dependent Plasticity with Application 
#      to Biofeedback
# 
#  Author: Dejan Pecevski, dejan@igi.tugraz.at
#
#  Date: February 2008
#
#================================================================================

import sys
import os

from pypcsim import *
from pypcsimplus import *
from pypcsimplus.common import *
sys.path.append('../packages/reward_gen/build')
from pyreward_gen import *
from numpy import *
import random, getopt
from datetime import datetime
from math import *
from tables import *
from mpi4py import MPI
import operator
import numpy
from matplotlib.mlab import find as find

def experiment(exp_run_name, expname = None, params={}):
    print "experiment run name: ", exp_run_name

    p = Parameters()
    
    script_container = ScriptContainer()
    script_container.loadScripts([sys.argv[0], 'figure_draft_journal.py'])

    ###################################################################
    # Global parameter values
    ###################################################################
    
    p.nNeurons        = 4000    # number of neurons
    p.minDelay        = 1e-3    # minimum synapse delay [sec]
    p.maxDelay = 1
    p.ConnP           = 0.02     # connectivity probability
    p.Frac_EXC        = 0.8     # fraction of excitatory neurons
    
    p.Tsim            = 1200    # duration of the simulation [sec]
    p.DTsim           = 1e-4    # simulation time step [sec]
        
    p.nSynRecordedNeurons = 50  # number of neurons to record the synapse weights from
    
    p.Frac_OU = 0.5
    p.Frac_OU_Inh = 0.5    
    p.ouConnP = 0.4 * p.ConnP
    
    p.lowOUScale = 0.2
    
    p.firingRate = 5
    
    p.WExcLowOUScale  = 1.0
    
    p.WLowOUScale = 1.0
            
    # weight scaling parameters
    p.Wscale = 0.8
    p.WExcScale = 1.0 
    p.WInhScale = 1.4
    p.WHighOUScale = 1.0
    
    # neuron parameters
    p.Rm = 1e8
    p.Cm = 30e-11
    p.Vthresh=-59.0e-3
    p.Vresting=-70e-3
    p.Vreset=-70e-3 
    p.Trefract=5e-3    
    p.Vinit=-70e-3     
    # other synapse parameters
    p.synTau = 5e-3
    p.synDelay = 1e-3     
    p.Delay_Heter = 0.0
    p.ErevExc = 0e-3
    p.ErevInh = -75e-3
    p.reinforced_nrn_perm_idx = 193
    
    p.UDF_Heter = 0.5   
        
    
    # Dynamic Synapse parameters
    p.createNodes([ 'EE', 'EI', 'IE', 'II' ])
    
    p.EE.U = 0.5
    p.EE.D = 1.1
    p.EE.F = 0.02
    p.EE.synDelay = 1e-3
    
    p.EI.U = 0.25
    p.EI.D = 0.7
    p.EI.F = 0.02
    
    p.IE.U = 0.05
    p.IE.D = 0.125
    p.IE.F = 1.2
    
    p.II.U = 0.32
    p.II.D = 0.144
    p.II.F = 0.06

    
    p.EE.Cscale = 1.0
    p.EI.Cscale = 1.0
    p.IE.Cscale = 1.2
    p.II.Cscale = 0.8

        
    p.pyRandomSeed = 232511
    p.simulationRNGSeed = 684342
    p.constructionRNGSeed = 15233571
    p.numpyRndSeed = 210592831
    
    
    # STDP parameters
    p.alpha = 1.05
    p.MuPos = p.MuNeg = 0.0
    p.stdpA = 0.001
    p.stdpTaupos = 30e-3
    p.stdpTauneg = 30e-3
    
    
    # eligibility trace parameters
    p.DAStdpRate = 0.3
    p.DATraceTau = 0.4
    #p.DATraceDelay = 0.0
    p.DATraceShape = 'alpha'
    
    p.rewardDuration = 1
    
    # reward parameters
    p.rewardDelay = 0.5  
    
    p.posAlphaDelay = 0.2
    p.rateTau = 0.2
    
    p.negAlphaDelay = 0.2
    p.negTau = 1
    
    p.rewardScale = 0.005
    
    p.Inoise = 0e-10
    
    p.idleT = 1.5

    p.override(params)
                
    p.Tinp = p.Tsim  # length of the initial stimulus [sec]
    
    p.synTauInh = p.synTau
    
    tau_m = p.Cm * p.Rm
    tau_s = p.synTau
    p.Wexc = ((p.Vthresh - p.Vinit) * p.WExcScale * p.Wscale)/ ((p.ErevExc - p.Vinit) * p.Rm * tau_s / (tau_m - tau_s) *  ((tau_s / tau_m) ** (tau_s / (tau_m - tau_s)) - (tau_s / tau_m) ** (tau_m / (tau_m - tau_s))))   
    
    tau_s = p.synTauInh    
    p.Winh = ((p.Vthresh - p.Vinit) * p.WInhScale * p.Wscale)/ ((p.Vinit - p.ErevInh) * p.Rm * tau_s / (tau_m - tau_s) *  ((tau_s / tau_m) ** (tau_s / (tau_m - tau_s)) - (tau_s / tau_m) ** (tau_m / (tau_m - tau_s))))
    
    p.WexcHighOU = p.Wexc * p.WHighOUScale
    p.WinhHighOU = p.Winh * p.WHighOUScale
    
    p.WexcLowOU = p.Wexc * p.WLowOUScale * p.WExcLowOUScale
    p.WinhLowOU = p.Winh * p.WLowOUScale
    
    
    print "average number of excitatory synapses = ", p.ConnP * p.nNeurons * p.Frac_EXC
        
    print "Wexc = ", p.Wexc, "Winh=", p.Winh
    
    p.stdpApos = p.stdpA * p.Wexc
    p.stdpAneg = - p.stdpA * p.alpha * p.Wexc
    
    p.nExcNeurons = int(p.nNeurons * p.Frac_EXC)
    p.nInhNeurons = p.nNeurons - p.nExcNeurons
    
    p.ouExcNeurons = int(p.nExcNeurons * p.Frac_OU)
    
    p.ouInhNeurons = int(p.nInhNeurons * p.Frac_OU_Inh)
    
    tstart=datetime.today()
    
    # init seeds 
    random.seed(datetime.today().microsecond)
    random.seed(p.pyRandomSeed)    
    numpy.random.seed(p.numpyRndSeed)
    
    p.samplingTime = int(p.Tsim / (200 * p.DTsim))  # sampling time for the recorded analog values
    
    def sub_time(t1, t2):
        return (t1 - t2).seconds+1e-6*(t1-t2).microseconds;
    
    ###################################################################
    # Create an empty network
    ###################################################################
    sp = SimParameter(dt=Time.sec(p.DTsim) , minDelay = Time.sec(p.minDelay), maxDelay = Time.sec(p.maxDelay), simulationRNGSeed = p.simulationRNGSeed, constructionRNGSeed = p.constructionRNGSeed)
    net = DistributedSingleThreadNetwork(sp)
    r = Recordings(net)
    
    ###################################################################
    # Create the neurons and set their parameters
    ###################################################################
    da_lifmodel = DARecvCbLifNeuron(Cm=p.Cm, Rm=p.Rm, Vthresh=p.Vthresh, Vresting=p.Vresting, Vreset=p.Vreset, Trefract=p.Trefract, Vinit=p.Vinit, Inoise = p.Inoise);
    
    exc_nrn_popul = SimObjectPopulation(net, da_lifmodel, int(p.nNeurons * p.Frac_EXC));
        
    inh_nrn_popul = SimObjectPopulation(net, da_lifmodel, p.nNeurons - exc_nrn_popul.size());
    
    all_nrn_popul = SimObjectPopulation(net, list(exc_nrn_popul.idVector()) + list(inh_nrn_popul.idVector()));
    
    
    
    #--------------------------------------------------------------------------------------------------
    the_permutation = numpy.random.permutation(p.nExcNeurons)
    
    r.exc_ou_nrn_idxs = the_permutation[:p.ouExcNeurons]
    
    r.exc_other_nrn_idxs = the_permutation[p.ouExcNeurons:]
    
    p.reinforced_nrn_idx = r.exc_other_nrn_idxs[p.reinforced_nrn_perm_idx]      
    
    r.exc_ou_nrn_ids = []
    for idx in r.exc_ou_nrn_idxs:
        r.exc_ou_nrn_ids.append(exc_nrn_popul[idx])
                                  
    exc_ou_nrn_popul = SimObjectPopulation(net, r.exc_ou_nrn_ids)
    
    r.exc_other_nrn_ids = []
    for idx in r.exc_other_nrn_idxs:
        r.exc_other_nrn_ids.append(exc_nrn_popul[idx])
    
    exc_other_nrn_popul = SimObjectPopulation(net, r.exc_other_nrn_ids)
    
    assert( p.reinforced_nrn_idx not in r.exc_ou_nrn_idxs )
            
    the_permutation = numpy.random.permutation(p.nInhNeurons)
    r.inh_ou_nrn_idxs = the_permutation[:p.ouInhNeurons]
    r.inh_other_nrn_idxs = the_permutation[p.ouInhNeurons:]
    
    r.inh_ou_nrn_ids = []
    for idx in r.inh_ou_nrn_idxs:
        r.inh_ou_nrn_ids.append(inh_nrn_popul[idx])
    
    r.inh_other_nrn_ids = []
    for idx in r.inh_other_nrn_idxs:
        r.inh_other_nrn_ids.append(inh_nrn_popul[idx])
    
    inh_ou_nrn_popul = SimObjectPopulation(net, r.inh_ou_nrn_ids)
    
    inh_other_nrn_popul = SimObjectPopulation(net, r.inh_other_nrn_ids)
    
    ou_nrn_popul = SimObjectPopulation(net, r.exc_ou_nrn_ids + r.inh_ou_nrn_ids)
    
    other_nrn_popul = SimObjectPopulation(net, r.exc_other_nrn_ids + r.inh_other_nrn_ids)
    
    net.mount(OUNoiseSynapse(0.012e-6, 0.003e-6, 2.7e-3, 0.0), ou_nrn_popul.idVector())
    net.mount(OUNoiseSynapse(0.057e-6, 0.0066e-6, 10.5e-3,-75e-3), ou_nrn_popul.idVector())
    
    net.mount(OUNoiseSynapse(0.012e-6 * p.lowOUScale, 0.003e-6 * p.lowOUScale, 2.7e-3, 0.0), other_nrn_popul.idVector())
    net.mount(OUNoiseSynapse(0.057e-6 * p.lowOUScale, 0.0066e-6 * p.lowOUScale, 10.5e-3,-75e-3), other_nrn_popul.idVector())
    
    print "Created", exc_nrn_popul.size(), "exc and", inh_nrn_popul.size(), "inh neurons";
    
    if p.DATraceShape == 'exp':
        DATraceResponse = ExponentialDecaySpikeResponse(p.DATraceTau)
    else:
        DATraceResponse = AlphaFunctionSpikeResponse(p.DATraceTau)
    
    ###################################################################
    # Create synaptic connections
    ###################################################################
    print 'Making synaptic connections:'
    t0=datetime.today()
    
    EE, EI, IE, II = 0, 1, 2, 3
    
    SynFactory = [EE, EI, IE, II]
    
    SynFactory[EE] = SimObjectVariationFactory(DAModStdpDynamicCondExpSynapse(Winit= p.WexcLowOU, 
                                                                                  Erev = p.ErevExc, 
                                                                                  tau = p.synTau, 
                                                                                  delay = p.synDelay, 
                                                                                  U = p.EE.U, 
                                                                                  D = p.EE.D, 
                                                                                  F = p.EE.F, 
                                                                                  activeDASTDP = True, 
                                                                                  Apos = p.stdpApos, 
                                                                                  Aneg = p.stdpAneg, 
                                                                                  taupos = p.stdpTaupos, 
                                                                                  tauneg = p.stdpTauneg, 
                                                                                  muneg = p.MuPos, 
                                                                                  mupos = p.MuNeg, 
                                                                                  Wex = 2.0 * p.WexcLowOU, 
                                                                                  useFroemkeDanSTDP = False,
                                                                                  DAStdpRate = p.DAStdpRate, 
                                                                                  daTraceResponse = DATraceResponse))
    
    SynFactory[EE].set("U", BndNormalDistribution(p.EE.U, p.UDF_Heter, 0.05, 0.95))
    SynFactory[EE].set("D", BndNormalDistribution(p.EE.D, p.UDF_Heter, 5e-3, 5))
    SynFactory[EE].set("F", BndNormalDistribution(p.EE.F, p.UDF_Heter, 5e-3, 5))    

    
    SynFactory[EI] = SimObjectVariationFactory(DynamicCondExpSynapse(W = p.WexcLowOU, 
                                                                     Erev = p.ErevExc, 
                                                                     tau = p.synTau, 
                                                                     delay = p.synDelay, 
                                                                     U = p.EI.U, 
                                                                     D = p.EI.D, 
                                                                     F = p.EI.F))
    
    SynFactory[EI].set("U", BndNormalDistribution(p.EE.U, p.UDF_Heter, 0.05, 0.95))
    SynFactory[EI].set("D", BndNormalDistribution(p.EE.D, p.UDF_Heter, 5e-3, 5))
    SynFactory[EI].set("F", BndNormalDistribution(p.EE.F, p.UDF_Heter, 5e-3, 5))    

    
    SynFactory[IE] = SimObjectVariationFactory(DynamicCondExpSynapse(W= p.WinhLowOU, 
                                                                      Erev = p.ErevInh, 
                                                                      tau = p.synTauInh, 
                                                                      delay = p.synDelay, 
                                                                      U = p.IE.U, 
                                                                      D = p.IE.D, 
                                                                      F = p.IE.F))
    
    SynFactory[IE].set("U", BndNormalDistribution(p.EE.U, p.UDF_Heter, 0.05, 0.95))
    SynFactory[IE].set("D", BndNormalDistribution(p.EE.D, p.UDF_Heter, 5e-3, 5))
    SynFactory[IE].set("F", BndNormalDistribution(p.EE.F, p.UDF_Heter, 5e-3, 5))    

    
    SynFactory[II] = SimObjectVariationFactory(DynamicCondExpSynapse(W = p.WinhLowOU, 
                                                                      Erev = p.ErevInh, 
                                                                      tau = p.synTauInh, 
                                                                      delay = p.synDelay, 
                                                                      U = p.II.U, 
                                                                      D = p.II.D, 
                                                                      F = p.II.F))
    
    SynFactory[II].set("U", BndNormalDistribution(p.EE.U, p.UDF_Heter, 0.05, 0.95))
    SynFactory[II].set("D", BndNormalDistribution(p.EE.D, p.UDF_Heter, 5e-3, 5))
    SynFactory[II].set("F", BndNormalDistribution(p.EE.F, p.UDF_Heter, 5e-3, 5))    

    
    syn_project = [EE, EI, IE, II]

    syn_project[EE] = ConnectionsProjection(exc_nrn_popul, exc_other_nrn_popul, 
                                                   SynFactory[EE], 
                                                   RandomConnections(conn_prob = p.ConnP * p.EE.Cscale), 
                                                   SimpleAllToAllWiringMethod(net), 
                                                   True, True)
    
    syn_project[EI] = ConnectionsProjection(exc_nrn_popul, inh_other_nrn_popul, 
                                                   SynFactory[EI], 
                                                   RandomConnections(conn_prob = p.ConnP * p.EI.Cscale), 
                                                   SimpleAllToAllWiringMethod(net), 
                                                   True, True)
    
    syn_project[IE] = ConnectionsProjection(inh_nrn_popul, exc_other_nrn_popul, 
                                                   SynFactory[IE], 
                                                   RandomConnections(conn_prob = p.ConnP * p.IE.Cscale))
    
    syn_project[II] = ConnectionsProjection(inh_nrn_popul, inh_other_nrn_popul, 
                                                   SynFactory[II], 
                                                   RandomConnections(conn_prob = p.ConnP * p.II.Cscale))
    
    
    
    # project to OU neurons with smaller connection probability
    OUSynFactory = [EE, EI, IE, II]
    
    OUSynFactory[EE] = SimObjectVariationFactory(DAModStdpDynamicCondExpSynapse(Winit= p.WexcHighOU, 
                                                                                  Erev = p.ErevExc, 
                                                                                  tau = p.synTau, 
                                                                                  delay = p.synDelay, 
                                                                                  U = p.EE.U, 
                                                                                  D = p.EE.D, 
                                                                                  F = p.EE.F, 
                                                                                  activeDASTDP = True, 
                                                                                  Apos = p.stdpApos, 
                                                                                  Aneg = p.stdpAneg, 
                                                                                  taupos = p.stdpTaupos, 
                                                                                  tauneg = p.stdpTauneg, 
                                                                                  muneg = p.MuPos, 
                                                                                  mupos = p.MuNeg, 
                                                                                  Wex = 2.0 * p.WexcHighOU, 
                                                                                  useFroemkeDanSTDP = False,
                                                                                  DAStdpRate = p.DAStdpRate, 
                                                                                  daTraceResponse = DATraceResponse))
    
    SynFactory[EE].set("U", BndNormalDistribution(p.EE.U, p.UDF_Heter, 0.05, 0.95))
    SynFactory[EE].set("D", BndNormalDistribution(p.EE.D, p.UDF_Heter, 5e-3, 5))
    SynFactory[EE].set("F", BndNormalDistribution(p.EE.F, p.UDF_Heter, 5e-3, 5))    

    
    OUSynFactory[EI] = SimObjectVariationFactory(DynamicCondExpSynapse(W = p.WexcHighOU, 
                                                                     Erev = p.ErevExc, 
                                                                     tau = p.synTau, 
                                                                     delay = p.synDelay, 
                                                                     U = p.EI.U, 
                                                                     D = p.EI.D, 
                                                                     F = p.EI.F))
    
    SynFactory[EI].set("U", BndNormalDistribution(p.EE.U, p.UDF_Heter, 0.05, 0.95))
    SynFactory[EI].set("D", BndNormalDistribution(p.EE.D, p.UDF_Heter, 5e-3, 5))
    SynFactory[EI].set("F", BndNormalDistribution(p.EE.F, p.UDF_Heter, 5e-3, 5))    
    
        
    OUSynFactory[IE] = SimObjectVariationFactory(DynamicCondExpSynapse(W= p.WinhHighOU, 
                                                                      Erev = p.ErevInh, 
                                                                      tau = p.synTauInh, 
                                                                      delay = p.synDelay, 
                                                                      U = p.IE.U, 
                                                                      D = p.IE.D, 
                                                                      F = p.IE.F))
    
    SynFactory[IE].set("U", BndNormalDistribution(p.EE.U, p.UDF_Heter, 0.05, 0.95))
    SynFactory[IE].set("D", BndNormalDistribution(p.EE.D, p.UDF_Heter, 5e-3, 5))
    SynFactory[IE].set("F", BndNormalDistribution(p.EE.F, p.UDF_Heter, 5e-3, 5))    
    
    
    
    OUSynFactory[II] = SimObjectVariationFactory(DynamicCondExpSynapse(W = p.WinhHighOU, 
                                                                      Erev = p.ErevInh, 
                                                                      tau = p.synTauInh, 
                                                                      delay = p.synDelay, 
                                                                      U = p.II.U, 
                                                                      D = p.II.D, 
                                                                      F = p.II.F))
    
    SynFactory[II].set("U", BndNormalDistribution(p.EE.U, p.UDF_Heter, 0.05, 0.95))
    SynFactory[II].set("D", BndNormalDistribution(p.EE.D, p.UDF_Heter, 5e-3, 5))
    SynFactory[II].set("F", BndNormalDistribution(p.EE.F, p.UDF_Heter, 5e-3, 5))    
    
        
    ou_syn_project = [EE,EI,IE,II] 
    
    ou_syn_project[EE] = ConnectionsProjection(exc_nrn_popul, exc_ou_nrn_popul, 
                                                   OUSynFactory[EE], 
                                                   RandomConnections(conn_prob = p.ouConnP * p.EE.Cscale), 
                                                   SimpleAllToAllWiringMethod(net), 
                                                   True, True)
    
    ou_syn_project[EI] = ConnectionsProjection(exc_nrn_popul, inh_ou_nrn_popul, 
                                                   OUSynFactory[EI], 
                                                   RandomConnections(conn_prob = p.ouConnP * p.EI.Cscale), 
                                                   SimpleAllToAllWiringMethod(net), 
                                                   True, True)
    
    ou_syn_project[IE] = ConnectionsProjection(inh_nrn_popul, exc_ou_nrn_popul, 
                                                   OUSynFactory[IE], 
                                                   RandomConnections(conn_prob = p.ouConnP * p.IE.Cscale))
    
    ou_syn_project[II] = ConnectionsProjection(inh_nrn_popul, inh_ou_nrn_popul, 
                                                   OUSynFactory[II], 
                                                   RandomConnections(conn_prob = p.ouConnP * p.II.Cscale))
    
    
    
    t1= datetime.today();
    print 'Created', int(syn_project[EE].size() + syn_project[EI].size() + syn_project[IE].size() + syn_project[II].size()), 'conductance based synapses in', (t1 - t0).seconds, 'seconds'
    
                                                             
    ###########################################################
    # Create the reward generator 
    ###########################################################
    reward_gen_id = net.add(RewardGenerator2(), SimEngine.ID(0, 0))
    
    pos_rate_syn = net.create(StaticCurrAlphaSynapse(p.rewardScale/(p.rateTau * exp(1.0)), p.rateTau, delay = 0), SimEngine.ID(0, 0))
    neg_rate_syn = net.create(StaticCurrAlphaSynapse(- p.rewardScale/(p.negTau * exp(1.0)), p.negTau, delay = 0), SimEngine.ID(0, 0))
    
    for i in range(all_nrn_popul.size()):
      net.connect(reward_gen_id, 0, all_nrn_popul[i], "DA_concentration", Time.ms(1))
        
    net.connect(pos_rate_syn, 0, reward_gen_id, 1, Time.ms(0))
    net.connect(neg_rate_syn, 0, reward_gen_id, 2, Time.ms(0))
    
    net.connect(all_nrn_popul[p.reinforced_nrn_idx], 0, pos_rate_syn, 0, Time.sec(p.posAlphaDelay))
    net.connect(all_nrn_popul[p.reinforced_nrn_idx], 0, neg_rate_syn, 0, Time.sec(p.negAlphaDelay))
    
    r.rate_syn = net.record(pos_rate_syn, AnalogRecorder(p.samplingTime))
    
    if p.Tsim <= 300:
        r.reward = net.record(reward_gen_id, AnalogRecorder())
    else:
        r.reward = net.record(reward_gen_id, AnalogRecorder(p.samplingTime))
    
    
    # ********************************************************************************************
    # SPIKE RECORDINGS
    
    ###########################################################
    # Create recorders to record spikes and voltage traces
    ###########################################################
    r.spikes = all_nrn_popul.record(SpikeTimeRecorder())
    
    #**********************************************************************************************
    
    #************************************************************************
    # GROUPS OF SYNAPSES
    # store the whole topology of the network in recordings
    
    r.to_other_EE_pairs = [ [syn_project[EE].prePostPair(i)[0].packed(),syn_project[EE].prePostPair(i)[1].packed()] for i in range(syn_project[EE].size()) ]
    r.to_other_EI_pairs = [ [syn_project[EI].prePostPair(i)[0].packed(),syn_project[EI].prePostPair(i)[1].packed()] for i in range(syn_project[EI].size()) ]
    r.to_other_IE_pairs = [ [syn_project[IE].prePostPair(i)[0].packed(),syn_project[IE].prePostPair(i)[1].packed()] for i in range(syn_project[IE].size()) ]
    r.to_other_II_pairs = [ [syn_project[II].prePostPair(i)[0].packed(),syn_project[II].prePostPair(i)[1].packed()] for i in range(syn_project[II].size()) ]
    
    r.to_ou_EE_pairs = [ [ou_syn_project[EE].prePostPair(i)[0].packed(),ou_syn_project[EE].prePostPair(i)[1].packed()] for i in range(ou_syn_project[EE].size()) ]
    r.to_ou_EI_pairs = [ [ou_syn_project[EI].prePostPair(i)[0].packed(),ou_syn_project[EI].prePostPair(i)[1].packed()] for i in range(ou_syn_project[EI].size()) ]
    r.to_ou_IE_pairs = [ [ou_syn_project[IE].prePostPair(i)[0].packed(),ou_syn_project[IE].prePostPair(i)[1].packed()] for i in range(ou_syn_project[IE].size()) ]
    r.to_ou_II_pairs = [ [ou_syn_project[II].prePostPair(i)[0].packed(),ou_syn_project[II].prePostPair(i)[1].packed()] for i in range(ou_syn_project[II].size()) ]
    
    
    
    # split circuit synapses into reinforced and non-reinforced group    
    r.reinforced_ou_nrn_syns, r.reinforced_ou_nrn_syns_idx, other_ou_nrn_syns = collect_synids_nrn(syn_project[EE], 
                                                                            exc_ou_nrn_popul, exc_nrn_popul, p.reinforced_nrn_idx)
    
    r.reinforced_other_nrn_syns, r.reinforced_other_nrn_syns_idx, other_other_nrn_syns = collect_synids_nrn(syn_project[EE], 
                                                                             exc_other_nrn_popul, exc_nrn_popul, p.reinforced_nrn_idx)
    
    syn_record_nrn_idxs = numpy.random.permutation(p.nNeurons)[:p.nSynRecordedNeurons]
    
    syn_record_nrn_idxs = syn_record_nrn_idxs + (syn_record_nrn_idxs >= p.reinforced_nrn_idx)
    
    r.recorded_other_not_ou_circ_syns, recorded_other_not_ou_circ_syns_idxs, the_rest_circ_syns = collect_synids_nrn(syn_project[EE], all_nrn_popul, all_nrn_popul, syn_record_nrn_idxs)
    
    r.recorded_other_ou_circ_syns, recorded_other_ou_circ_syns_idxs, the_rest_circ_syns = collect_synids_nrn(ou_syn_project[EE], all_nrn_popul, all_nrn_popul, syn_record_nrn_idxs)
    
    r.recorded_other_circ_syns = r.recorded_other_ou_circ_syns + r.recorded_other_not_ou_circ_syns
    
    r.exc_ou_afferents_reinforced_nrn = collect_afferents(syn_project[EE], exc_ou_nrn_popul, exc_nrn_popul, p.reinforced_nrn_idx)
    
    r.exc_other_afferents_reinforced_nrn = collect_afferents(syn_project[EE], exc_other_nrn_popul, exc_nrn_popul, p.reinforced_nrn_idx)
    
    r.exc_afferents_reinforced_nrn = r.exc_ou_afferents_reinforced_nrn + r.exc_other_afferents_reinforced_nrn
    
    print "num of reinforced synapses = ", len(r.reinforced_ou_nrn_syns) + len(r.reinforced_other_nrn_syns)
    
    #*************************************************************************************
    # WEIGHTS RECORDINGS
    
    
    r.reinforced_ou_weights = SimObjectPopulation(net, r.reinforced_ou_nrn_syns).record(AnalogRecorder(p.samplingTime), "W")
    r.reinforced_other_weights = SimObjectPopulation(net, r.reinforced_other_nrn_syns).record(AnalogRecorder(p.samplingTime), "W")
    
    r.other_circ_not_ou_weights = SimObjectPopulation(net, r.recorded_other_not_ou_circ_syns).record(AnalogRecorder(p.samplingTime), "W")
    r.other_circ_ou_weights = SimObjectPopulation(net, r.recorded_other_ou_circ_syns).record(AnalogRecorder(p.samplingTime), "W")    
    
    ###################################################
    # Number of synapses report
    ###################################################
    syntype_str = ['EE','EI','IE','II']
    p.numSyn = [EE,EI,IE,II]
    
    for i in [EE,EI,IE,II]:
        len_project = syn_project[i].size()
        len_ou_project = ou_syn_project[i].size()
        
        print "number of %s synapses lowOU:" % (syntype_str[i],), len_project
        print "number of %s synapses highOU" % (syntype_str[i],), len_ou_project
        print "total number of %s synapses " %  (syntype_str[i],), len_project + len_ou_project
        p.numSyn[i] =  len_project + len_ou_project
        
    print "total number of excitatory synapses : ", syn_project[EE].size() + syn_project[EI].size() + ou_syn_project[EE].size() + ou_syn_project[EI].size()
    p.totalNumExcSyn = syn_project[EE].size() + syn_project[EI].size() + ou_syn_project[EE].size() + syn_project[EI].size()
    print "total number of synapses : ", syn_project[EE].size() + syn_project[EI].size() + syn_project[IE].size() + syn_project[II].size() + ou_syn_project[EE].size() + ou_syn_project[EI].size() + ou_syn_project[IE].size() + ou_syn_project[II].size()
    p.totalNumExcSyn = syn_project[EE].size() + syn_project[EI].size() + syn_project[IE].size() + syn_project[II].size() + ou_syn_project[EE].size() + ou_syn_project[EI].size() + ou_syn_project[IE].size() + ou_syn_project[II].size()
    
     
        
    def set_learning(proj_list, new_state):
        for proj in proj_list:            
            for i in range(proj.size()):       
                if proj.object(i) != None:           
                    proj.object(i).activeDASTDP = new_state
    
    
    ############################################################
    # SIMULATE THE CIRCUIT
    ############################################################
    print 'Running simulation:';
    t0=datetime.today()
    
    net.add(SimProgressBar(Time.sec(p.Tsim)), SimEngine.ID(0, 0))
    
    print "Simulation start: " , datetime.today().strftime('%x %X')
    net.reset();
    
    set_learning([syn_project[EE], ou_syn_project[EE]], False)
    
    net.advance(int(p.idleT/p.DTsim))
    
    set_learning([syn_project[EE], ou_syn_project[EE]], True)
    
    net.advance(int((p.Tsim - p.idleT)/p.DTsim))
    
    
    t1=datetime.today()
    print 'Done.', (t1-t0).seconds, 'sec CPU time for', p.Tsim*1000, 'ms simulation time';
    print '==> ', (t1-tstart).seconds, 'seconds total'
    p.simDuration = (t1-t0).seconds
    p.numProcesses = net.mpi_size()
            
    print "Saving results..."
    
    exp_run_name = 'noname'
    if len(sys.argv) > 1:
        exp_run_name = sys.argv[1]
    
    if expname is None:
        expname = sys.argv[0]
    
    f = open_experiment_h5file(expname, exp_run_name)
        
    p.saveInH5File(f)    
    r.saveInOneH5File(f)
                
    script_container.storeScripts(f)
            
    if not f is None:
        print "closing file"
        f.close();
         
    print "Done."


if len(sys.argv) > 1:    
    experiment(sys.argv[1], None, {})    
else:    
    model_params["Tsim"] = 120.0
    experiment('noname', None, model_params)
    