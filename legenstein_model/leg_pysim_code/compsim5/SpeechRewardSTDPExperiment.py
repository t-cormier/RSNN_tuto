#======================================================================
#  Computer simulation 5 from
#      Legenstein, R., Pecevski, D. and Maass, W., A Learning Theory
#       for Reward-Modulated Spike-Timing-Dependent Plasticity with 
#       application to Biofeedback
# 
#  Author: Dejan Pecevski, dejan@igi.tugraz.at
#
#  Date: March 2008
#
#======================================================================

import sys
from sys import *
import os

import matplotlib
matplotlib.use('Agg')


from pypcsim import *
import pypcsimplus as pcsim
from pypcsimplus.clusterUtils import *
from pypcsimplus.RemoteExperiment import *

import getopt
import numpy
from datetime import datetime
from math import *
from tables import *
from math import exp
from mpi4py import MPI
from LiquidModel400 import *
from ReadoutModel import *
import IPython.kernel.client as kernel

import pygsl.rng
from pylab import *
from numpy import *
import pickle


class SpeechRewardSTDPExperiment(pcsim.Experiment):
        
    def defaultExpParameters(self):
        ep = self.expParams 
        
        # General simulation parameters        
        ep.DTsim = 1e-4
        
        ep.nTrainEpochs = 100
        ep.nTestEpochs = 10
        
        ep.initT = 100e-3
        
        ep.trialT = 2.5
        
        # Network distribution parameters
        ep.netType = 'ST'
        ep.nThreads = 3
        ep.minDelay = 1e-3
        ep.maxDelay = 2   
        
        # Seeds of the experiment
        ep.numpyRandomSeed = 31342
        ep.pyRandomSeed = 124243        
        ep.constructionSeed = 3224356
        ep.simulationSeed = 134252439
        
        ep.spikeTemplatesSeed = 13046346
        ep.testSpikesSeed = 60517
        
        ep.pyGSLSeed = 1623235635
        ep.runMode = "long"        
        ep.liquid = "LiquidModel400"
        
        ep.numIPEngines = 20
        
        ep.numTestsPerCase = 5
        
        ep.recordReadoutVm = True
        
        ep.numTrialsWithoutThreshold = 20
        ep.numTrialsRecordVm = 40
        
        ep.numRepetitionsTest = 10

        ep.ipcontroller_host = 'no default'
        
    
    def prepareCluster(self):
        ep = self.expParams
        
    
        IPcontroller = {'host' : ep.ipcontroller_host, 'engine_port' : 32100, 'rc_port' : 32101, 'task_port' : 32102}        
        self.IPcluster = Cluster(ClusterConfig(configFile= './clusterconf.py', controller=IPcontroller))
        self.IPcluster.start(waitafter = 4)
        self.IPcluster.connect()
        rc = self.IPcluster.getRemoteControllerClient()
        
        print "len(rc) is ", len(rc)
        while len(rc) < 30:
            print "len(rc) is ", len(rc)
            time.sleep(1)
        
        
        rc.execute('import os,sys')
        rc.execute('import pypcsimplus')
        rc.execute('from pypcsimplus import *')    
        rc.execute('import pypcsimplus.common')   
        pwd = os.getcwd()     
        rc.execute('os.chdir("' + pwd + '")')
        rc.execute('sys.path.append("")')    
        rc.execute('import Liquid400Experiment as Ex')
        print "The cluster is setup"
        return rc
    
        
    def setupModels(self):        
        p = self.modelParams
        ep = self.expParams
        m = self.models
        net = self.net
        
        numpy.random.seed(ep.numpyRandomSeed)
        
        # setup Seeds
        gslRandom = pygsl.rng.gfsr4()
        gslRandom.set(ep.pyGSLSeed)

        rc = self.prepareCluster()
    
        m.remoteExps = [ RemoteExperiment(self.IPcluster, ipEngine) for ipEngine in rc.get_ids() ]
        
        ep.numpyIPEnginesSeed = []
        ep.simulationIPEnginesSeed = []
        for eid in range(len(m.remoteExps)): 
            ep.numpyIPEnginesSeed.append(gslRandom.get())
            ep.simulationIPEnginesSeed.append(gslRandom.get())
        
        for i in range(len(m.remoteExps)):
            liq_exp_params = {'runMode':ep.runMode}
            liq_exp_params["numpyRandomSeed"] = ep.numpyIPEnginesSeed[i]
            liq_exp_params["simulationSeed"] = ep.simulationIPEnginesSeed[i]
            liq_exp_params['ipEngineID'] = rc.get_ids()[i]    
            m.remoteExps[i].create("Ex.Liquid400Experiment", "liq_exper", experParams = liq_exp_params, modelParams = p.get('liquid', {}))
            m.remoteExps[i].setup()
        print "All remote experiments created"
        
        ep.liquid = m.remoteExps[0].getModel('liquid').params
        ep.liq_input = m.remoteExps[0].getModel('input').params
        
        ep.Tsim = ep.nTrainEpochs * ep.trialT
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
        
        rewInpParams = p.get("rewardInput", {}) 
        rewInpParams['nExcNeurons'] = ep.liquid.nExcNeurons
        rewInpParams['nInhNeurons'] = ep.liquid.nInhNeurons
        m.rewardInput = RewardInputModel(self.net, self.expParams, rewInpParams)
        
        m.rewardInput.generate()
                
        # create the readout model
        m.readout = ReadoutModel(self.net, self.expParams, p.get("readout", {}), depModels = m.rewardInput.elements)        
        m.readout.generate()
        
        m.rewardInput.connectReadout(m.readout)
        
        # prepare tuples (speaker, utterance, digit) (only digit 1 and 2)
        self.prepareSUDs()
        
        print "Models created"
        
        
    def setupRecordings(self):
        m = self.models
        r = self.recordings        
        r.rewardInput = self.models.rewardInput.setupRecordings()        
        r.readout = self.models.readout.setupRecordings()
        r.SudList = m.SudList
        r.sudListSegments = m.sudListSegments
        r.phaseNum = m.phaseNum
        return r
    
    
    def prepareRecordVmNoThreshold(self):
        m = self.models
        m.readout.deactivateLearning()
        m.readout.increaseThreshold()    
        
        
    def prepareRecordVm(self):
        m = self.models              
        m.readout.setNormalThreshold()
    
        
    def preparePretrain(self):
        m = self.models
        m.readout.switchOffRecordVmReadout()
    
    
    def prepareTrain(self):
        m = self.models
        m.readout.activateLearning()
        
    
    def prepareTest(self):
        m = self.models
        m.readout.deactivateLearning()
    
    
    def prepareRecordVm2(self):
        m = self.models
        m.readout.switchOnRecordVmReadout()
    
    
    def prepareRecordVmNoThreshold2(self):
        m = self.models              
        m.readout.increaseThreshold()
        
        
    def prepareSUDs(self):
        ep = self.expParams
        m = self.models
        m.speakers = [1, 2, 5, 6, 7]
        m.speakers = [2]
        
        m.sudListSegments = []
        m.phasePreparations = []        
        m.SudList = []
        
        m.phaseNum = {'recordVmNoThreshold':0,'recordVm':1,
                      'preTrain':2,'train':3,'test':4, 'recordVm2':5,'recordVmNoThreshold2':6}
        
        for dig in range(1,3):
            for utter in range(1,11):
                m.SudList.append((m.speakers[0], utter, dig))
        m.sudListSegments.append(('recordVmNoThreshold', 0, len(m.SudList)))
        m.phasePreparations.append(self.prepareRecordVmNoThreshold)
    
        
        firstTrial = len(m.SudList)
        for dig in range(1,3):
            for utter in range(1,11):
                m.SudList.append((m.speakers[0], utter, dig))
        m.sudListSegments.append(('recordVm', firstTrial, len(m.SudList)))
        m.phasePreparations.append(self.prepareRecordVm)
        
        
        # setup pretrain phase
        firstTrial = len(m.SudList) 
        for epoch in range(ep.numRepetitionsTest):
            for dig in range(1,3):
                for utter in range(1,11):
                    m.SudList.append((m.speakers[0], utter, dig))            
        m.sudListSegments.append(('preTrain',firstTrial,len(m.SudList)))
        m.phasePreparations.append(self.preparePretrain)
        
        
        # setup training phase
        firstTrial = len(m.SudList)
        currDigit = 1
        for epoch in range(ep.nTrainEpochs):            
            randomUtter = random.randint(1, 11)         
            m.SudList.append((m.speakers[random.randint(0, len(m.speakers))], randomUtter, currDigit))
            currDigit = currDigit % 2 + 1
        m.sudListSegments.append(('training',firstTrial,len(m.SudList)))
        m.phasePreparations.append(self.prepareTrain)
        
        firstTrial = len(m.SudList)         
        for epoch in range(ep.numRepetitionsTest):
            for dig in range(1,3):
                for utter in range(1,11):
                    m.SudList.append((m.speakers[0], utter, dig))
        m.sudListSegments.append(('test', firstTrial, len(m.SudList)))
        m.phasePreparations.append(self.prepareTest)
        
            
        firstTrial = len(m.SudList)
        for dig in range(1,3):
            for utter in range(1,11):
                m.SudList.append((m.speakers[0], utter, dig))         
        m.sudListSegments.append(('recordVm2', firstTrial, len(m.SudList)))
        m.phasePreparations.append(self.prepareRecordVm2)
        
        firstTrial = len(m.SudList)
        for dig in range(1,3):
            for utter in range(1,11):
                m.SudList.append((m.speakers[0], utter, dig))         
        m.sudListSegments.append(('recordVmNoThreshold2', firstTrial, len(m.SudList)))
        m.phasePreparations.append(self.prepareRecordVmNoThreshold2)
        
    
    
    def simulate(self):
        ep = self.expParams        
        ep.samplingTime = int(ep.Tsim / (200 * ep.DTsim))
        m = self.models
        r = self.recordings
        
        # Run simulation 
        print 'Running lsms:', datetime.today().strftime('%x %X')
        
        t0=datetime.today()
        
        self.net.reset();
        
        cmd = """
exper.reset()
exper.models.input.resetStimulus(SUD)
exper.run('oneRun', saveData = False)
resp,inp = exper.getOutput()
        """
        
        tc = self.IPcluster.getTaskControllerClient()
        
        tids=[]
        print "Preparing liquid responses for training phase..."
        for i in range(len(m.SudList)):        
            tids.append(tc.run(kernel.StringTask(cmd, pull=['resp','inp'], push=dict(SUD=m.SudList[i]))))

        print len(tids), 'Tasks started...'
        tc.barrier(tids)
        
        m.trainResp = []
        m.trainInp = []
        for tid in range(len(tids)):
            res = tc.get_task_result(tids[tid])
            if None<>res.failure:
                res.failure.printDetailedTraceback()
                res.failure.raiseException()            
            m.trainResp.append(res.results['resp'])
            m.trainInp.append(res.results['inp'])
        print "Done."
        
        print "number of synapses of the first circuit is ", m.trainResp[0].numSynapses
        r.numSynapses = m.trainResp[0].numSynapses
        
        print "Shutting down cluster ..."
        self.IPcluster.stop()
        print "Done"
        
        currEpoch = 0        
        for currPhase in range(len(m.sudListSegments)):
            m.phasePreparations[currPhase]()
            print "Starting phase :", m.sudListSegments[currPhase][0]
            for epochWithinPhase in range(m.sudListSegments[currPhase][1],m.sudListSegments[currPhase][2]):
                if currEpoch % 10 == 0:
                    stdout.write(str(currEpoch))
                else:
                    stdout.write(".")
                            
                m.rewardInput.reset(currEpoch, m.trainResp[currEpoch], m.SudList[currEpoch][2])
                m.readout.diminishWeights()                                            
                self.net.advance(int(ep.trialT / ep.DTsim))            
                currEpoch  += 1
            print "done."
            
        
        t1=datetime.today()
        print '\nDone.', (t1-t0).seconds, 'sec CPU time for', ep.Tsim, 's simulation time';        
        self.expParams.simDuration = (t1 - t0).seconds
        
        
    def scriptList(self):
        return ["SpeechRewardSTDPExperiment.py", "Liquid400Experiment.py", "LiquidModel400.py" ,\
                'SpeechStimulusInputModel.py', 'StimulusModel.py' ]
    
if __name__ == "__main__":
    numpySeedArray = [1305835, 4609385, 143094, 10598, 16093815, 12509823, 1240, 4398, 2523, 205893, 
                      24098, 640968, 140981, 230498, 204982, 50985, 6145454, 24098, 509285, 1230981 ]
try:
    runName = "SpeechRMSTDP_final_"   
    seedNo = 1
    longRun = False
    # -- process options
    opts, args = getopt.getopt( sys.argv[1:], "s:e:l", [ "seedNo=", "runName=",'longRun','ipcontroller-host='] )
    for opt, arg in opts:
        if opt in ( "-s", "--seedNo" ):
            seedNo = arg
        elif opt in ('--runName'):
            runName = arg
        elif opt in ('-l', '--longrun'):
            longRun = True
        elif opt in ('--ipcontroller-host'):
            ipcontroller_host = arg
            
    if longRun:        
        exper = SpeechRewardSTDPExperiment('SpeechRewardSTDP', experParams = {"numpyRandomSeed" : numpySeedArray[seedNo],
                                                                              "ipcontroller_host" : ipcontroller_host, 
                                                                              "nTrainEpochs":2000, "nTestEpochs":30 })
        exper.setup()
        print exper.models.readout.params
        exper.run(runName+ "_" + sys.argv[1])
    else:
        # NOTE : nTrainEpochs and nTestEpochs MUST be even numbers        
        exper = SpeechRewardSTDPExperiment('SpeechRewardSTDP', experParams = {"numpyRandomSeed" : 354093825,    
                                        "spikeTemplatesSeed" : 5958,
                                        'ipcontroller_host' : ipcontroller_host, 
                                        "nTrainEpochs":2000, "nTestEpochs":40, "runMode" : "short" }, 
                                        modelParams ={ 'readout': {},    
                                                         'liquid': { 
                                                                'input': {},
                                                                'liquid': {} 
                                                            } })
        exper.run("final")
except e:
    print e

