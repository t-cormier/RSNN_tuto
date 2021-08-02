import utils
import logging
import numpy
import copy

from readout import *
from linear_classification import *
from inputs.jitteredtemplate import *
from targetfunction import *
from response2states import *

class Performance(object):
#    def __init__(self, cc=0.0, mae=0.0, mse=0.0, score=0.0, CM=None, mi=0.0, hx=0.0, hy=0.0, kappa=numpy.nan):
    def __init__(self, cc=numpy.nan, mae=numpy.nan, mse=numpy.nan, score=numpy.nan, CM=None, mi=numpy.nan, hx=numpy.nan, hy=numpy.nan, kappa=numpy.nan):
        if CM is None:
            CM=numpy.zeros((2,2))

        self.cc=cc
        self.mae=mae
        self.mse=mse
        self.score=score
        self.CM=CM
        self.mi=mi
        self.hx=hx
        self.hy=hy
        self.kappa=kappa

    def __str__(self):
        desc = """  PERFORMANCE
    cc=%s, mae=%s, mse=%s, score=%s, kappa=%s
    CM=%s
    mi=%s, hx=%s, hy=%s
    """ % (self.cc, self.mae, self.mse, self.score, self.kappa, self.CM, self.mi, self.hx, self.hy)
        return desc


def trainReadouts(readouts, states_train, stimulus_train=None, states_test=None, stimulus_test=None, 
                  targets_train=None, targets_test=None, sampling = None, raise_exceptions=True):
    '''
    '''
    perfTrain=[]
    perfTest=[]

    if len(states_train) == 0:
        raise Exception("training dataset is empty")

    if stimulus_train is None: stimulus_train = []
    if states_test is None: states_test=[]
    if stimulus_test is None: stimulus_test=[]

    logging.info("training readouts ...")
    r_i = 0
    for r in readouts:
        print "training readout %d" % (r_i,)
        r_i+=1 
        
        logging.debug('len(stimulus_train): %d, len(states_train): %d', len(stimulus_train), len(states_train))

        perf_train=Performance()
        try:
            (mae, mse, cc, score, CM, mi, hx, hy, kappa) = r.train(stimulus_train, states_train, targets_train, sampling=sampling)
            perf_train=Performance(cc, mae, mse, score, CM, mi, hx, hy, kappa)
        except Exception, inst:
            if raise_exceptions:
                raise inst

        logging.info("train performance: cc=%f, mae=%f, mse=%f, score=%f", perf_train.cc, perf_train.mae, perf_train.mse, perf_train.score)
        perfTrain.append(perf_train)

        perf_test=Performance()
        if len(states_test) > 0:
            try:
                logging.debug('len(stimulus_test): %d, len(states_test): %d', len(stimulus_test), len(states_test))
                (mae, mse, cc, score, CM, mi, hx, hy, kappa) = r.performance(stimulus_test, states_test, targets_test, sampling=sampling)
                logging.info("test performance: cc=%f, mae=%f, mse=%f, score=%f", cc, mae, mse, score)
                perf_test=Performance(cc, mae, mse, score, CM, mi, hx, hy, kappa)
            except Exception, inst:
                if raise_exceptions:
                    raise inst

        perfTest.append(perf_test)

    return (perfTrain, perfTest)



class ExternalReadout(Readout):
    '''
    '''
    def __init__(self, targetFunction=None, algorithm=None, doNorm=True, noise=0.0, 
                 Vpca=None, Kstratify=None, channels=None, neg_channels=None):
        '''
        '''
        if algorithm is None:
            algorithm = LinearClassification(addBias=True)

        self.algorithm = algorithm
        self.targetFunction = targetFunction
        self.doNorm = doNorm                  # True ... mean/std normalization, False ... no normalization
        self.noise = noise                    # Amount of noise added to the data before training
        self.channels = copy.copy(channels)   # channels to consider

        if neg_channels is None:
            self.num_neg = 0
        else:
            self.num_neg = len(neg_channels)
            if self.channels is None:
                self.channels = copy.copy(neg_channels)
            else:
                self.channels += copy.copy(neg_channels)

        if Vpca is None: Vpca = []
        self.Vpca = Vpca                      # The fraction of variance to hold in the data after a PCA
        self.Kstratify = Kstratify            # The #pos/#neg ratio of the stratified data.';


    def makeSet(self, stimulus, states, targets=None, sampling=None):
        '''
        '''
        if len(stimulus) != len(states):
            raise Exception("number of Stimuli does not correspond to number of states")

        ts_X=[]
        ts_Y=[]
        if isinstance(states, numpy.ndarray):
            ts_Y = self.makeSet_Y(stimulus, targets, sampling)
            if states.ndim == 3:
                flat_states = states.reshape(states.shape[0]*states.shape[1], states.shape[2])
            else:
                flat_states = states
            if not self.channels is None:
#               print 'states.shape:',states.shape
#               print 'len(self.channels):', len(self.channels)
                ts_X = copy.copy(flat_states[:, self.channels])
                if self.num_neg > 0:
                    ts_X[:,-self.num_neg:] *= -1
            else:
                ts_X = flat_states.copy()
        else:
#            (ts_X, ts_Y) = self.makeSet(stimulus, states, targets)

            for i in range(len(stimulus)):
                sampling=[state.t for state in states[i]]

                if not self.channels is None:
                    x=numpy.array([state.X[self.channels] for state in states[i]])
                    if self.num_neg > 0:
                        x[:,-self.num_neg:] *= -1
                else:
                    x=numpy.array([state.X for state in states[i]])

                ts_X.append(x)

                if targets is None:
                    if self.targetFunction is None:
                        raise Exception('No target function and no targets defined')
                    else:
                        y=self.targetFunction.values(stimulus[i], sampling)
                        ts_Y.append(y)

            ts_X=numpy.concatenate(ts_X)

            if targets is None:
                ts_Y=numpy.concatenate(ts_Y)
            else:
                ts_Y=numpy.array(targets)

        return (ts_X, ts_Y)


    def makeSet_Y(self, stimulus, targets=None, sampling = None):
        '''
        '''
        ts_Y=[]
        for i in range(len(stimulus)):
            if targets is None:
                if self.targetFunction is None:
                    raise Exception('No target function and no targets defined')
                else:
                    y=self.targetFunction.values(stimulus[i], sampling)
                    ts_Y.append(y)

        if targets is None:
            ts_Y=numpy.concatenate(ts_Y)
        else:
            ts_Y=numpy.array(targets)

        return ts_Y


    def stratify_data(self, ts_X, ts_Y):
        ''' stratify the data
        '''
        if self.Kstratify is not None:
            logging.info('Stratifying data (Kstratify=%d)...' % self.Kstratify)
            i_pos = numpy.where(ts_Y>0)[0].tolist()
            i_neg = numpy.where(ts_Y<=0)[0].tolist()
            if len(i_pos)==0 or len(i_neg)==0:
                logging.info("only training examples of one class!")
                return (ts_X, ts_Y)
            if self.Kstratify > 0:
                if self.Kstratify*len(i_pos) < len(i_neg):
                    i_kill = i_neg
                    i_hold = i_pos
                elif len(i_pos) > self.Kstratify*len(i_neg):
                    i_kill = i_pos
                    i_hold = i_neg
                else:
                    i_kill = []
                    i_hold = i_pos + i_neg
                if len(i_kill)>0:
                    maxl = max(len(i_pos), len(i_neg))
                    minl = min(len(i_pos), len(i_neg))
                    nk = max(maxl - self.Kstratify*minl, 0)
                    i_kill = numpy.random.permutation(i_kill).tolist()
                    i_hold += i_kill[nk:]
                    ts_X = ts_X[i_hold,:]
                    ts_Y = ts_Y[i_hold]
            elif self.Kstratify == -1:
                raise NotImplementedError("Kstratify==-1 not implemented yet")
            i_pos = numpy.where(ts_Y>0)[0]
            i_neg = numpy.where(ts_Y<=0)[0]
            if len(i_pos) > len(i_neg):
                logging.info("#pos/#neg = %s" % (float(len(i_pos))/float(len(i_neg))))
            else:
                logging.info("#neg/#pos = %s" % (float(len(i_neg))/float(len(i_pos))))

        return (ts_X, ts_Y)


    def train(self, stimulus, states, targets=None, sampling = None):
        '''train the external readout
        simulus ... list of Stimulus objects
        states ... list of States
        targets ... if targets given, the target function will be ignored
        '''
        (ts_X, ts_Y) = self.makeSet(stimulus, states, targets, sampling)
#        print 'ts_X.shape:',ts_X.shape,'ts_Y.shape:',ts_Y.shape
        (ts_X, ts_Y) = self.stratify_data(ts_X, ts_Y)

        logging.debug("ts_Y: %s", str(ts_Y))

        dim = ts_X.shape[1]
        logging.info("training with %d data points of dimension %d", len(ts_Y),dim)

        self.algorithm.train(ts_X, ts_Y)

        return self.algorithm.analyse(ts_X, ts_Y)


    def apply(self, X):
        '''
        '''
        return self.algorithm.apply(X)


    def performance(self, stimulus, states, targets=None, sampling = None):
        '''test the performance of the external readout
        simulus ... list of Stimulus objects
        states ... list of States
        '''
        (ts_X, ts_Y) = self.makeSet(stimulus, states, targets, sampling)

        logging.debug("ts_Y: %s", str(ts_Y))

        return self.algorithm.analyse(ts_X, ts_Y)


if __name__=='__main__':

    Tsim=1.0
    freq=20
    nChannels=5
    nTrials=40
    nTest=5
    nTrain=nTrials-nTest

    jtemp=JitteredTemplate(Tstim=Tsim, nChannels=nChannels, nTemplates=[2,2], jitter=4e-3, freq=[10,20])
    stimulus=[]
    states=[]
    for t in range(nTrials):
        stim=jtemp.generate()

        resp = Response(Tsim)
        for i in range(nChannels):
            resp.appendChannel(Channel(data=numpy.random.uniform(0,Tsim, freq*Tsim)))

        stimulus.append(stim)
#        states.append(response2states(resp, [Tsim/2, Tsim])[0])
        states.append(response2states(resp, [Tsim])[0])
    
    readout=ExternalReadout(SegmentClassification(posSeg=0))
    readout2=ExternalReadout(SegmentClassification(posSeg=1), channels=[1,3])
    readout3=ExternalReadout(SegmentClassification(posSeg=1), channels=[1,3], neg_channels=[2])
    readout4=ExternalReadout(SegmentClassification(posSeg=0), channels=range(nChannels))

    readouts=[readout, readout2, readout3, readout4]

    (mae, mse, cc, score, CM, mi, hx, hy, kappa) = readout.train(stimulus[:nTrain], states[:nTrain])
    print "1. train performance: cc=", cc, "mae=", mae, "mse=", mse, "score=", score, "kappa=", kappa
    (mae, mse, cc, score, CM, mi, hx, hy, kappa) = readout2.train(stimulus[:nTrain], states[:nTrain])
    print "2. train performance: cc=", cc, "mae=", mae, "mse=", mse, "score=", score, "kappa=", kappa
    (mae, mse, cc, score, CM, mi, hx, hy, kappa) = readout3.train(stimulus[:nTrain], states[:nTrain])
    print "3. train performance: cc=", cc, "mae=", mae, "mse=", mse, "score=", score, "kappa=", kappa
    (mae, mse, cc, score, CM, mi, hx, hy, kappa) = readout4.train(stimulus[:nTrain], states[:nTrain])
    print "4. train performance: cc=", cc, "mae=", mae, "mse=", mse, "score=", score, "kappa=", kappa

    readouts=[readout, readout2, readout3, readout4]

    perfTrain, perfTest=trainReadouts(readouts, states[:nTrain], stimulus[:nTrain], states[nTrain:], stimulus[nTrain:])
#    perfTrain, perfTest=trainReadouts(readouts, states[:nTrain], stimulus[:nTrain])

    class_seg0=SegmentClassification(posSeg=0)
    v=numpy.zeros(len(stimulus))
    for i in range(len(stimulus)):
        stim = stimulus[i]
        v[i]=class_seg0.values(stim, [Tsim])

    perfTrainValues, perfTestValues=trainReadouts([readout], states[:nTrain],
                                                  stimulus[:nTrain], states[nTrain:], stimulus[nTrain:], v[:nTrain], v[nTrain:])
