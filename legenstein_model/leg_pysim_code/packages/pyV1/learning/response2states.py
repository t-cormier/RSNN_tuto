import numpy
import sys
import logging

from pyV1.dataformats import *
from datetime import datetime

import pyV1.utils

def spikes2exp(spikes, sample_time, tau=30e-3, tau2=0):
    delta = (spikes.compress(spikes <= sample_time) - sample_time)/tau
    return (numpy.exp(delta)).sum()


def spikes2alpha(spikes, sample_time, tau1=30e-3, tau2=3e-3):
    MIN_EXP=-20
    delta = (spikes.compress(spikes <= sample_time) - sample_time)
    return (numpy.exp(delta/tau1)-numpy.exp(delta/tau2)).sum()


def spikes2count(spikes, sample_time, tau1=30e-3, tau2=0):
    delta = (spikes.compress(spikes <= sample_time) - sample_time)
    return (delta.compress(delta > -tau1)).shape[0]


class State(object):
    def __init__(self, values, t):
        self.X=values
        self.t=t


#@utils.timedTest
def response2states(Rin, sampling=None, filter=spikes2exp, tau1=30e-3, tau2=3e-3, channels=None):
    if utils.isScalar(Rin):
        Rin=[Rin]
        
    numResponses = len(Rin)
    all_states = []
    count = 0
    sys.stdout.write("response2states:   0%")
    for R in Rin:
        Tmax = R.Tsim
        if sampling is None: 
            sampling = [R.Tsim]
        if type(sampling)==str:
            sampling = eval("numpy.r_"+sampling)

        if channels is None:
            channels = range(len(R.channel))

        numChannels=len(channels)
        numSamplings=len(sampling)
        states = []

        if tau2 > tau1:
            tau1, tau2 = tau2, tau1

        logging.debug("Sampling %d %d-dimensional states...\n" % (numSamplings,numChannels))
        for sample_time in sampling:
            state = []
            for chid in channels:
                channel = R.channel[chid]
                channel_state = filter(numpy.atleast_1d(channel.data), sample_time, tau1, tau2)
                state.append(channel_state)
                count += 1
                sys.stdout.write("\b\b\b\b%3d%%" % (count*100/(numChannels*numSamplings*numResponses)))

            states.append(State(values=numpy.array(state), t=sample_time))
            
        all_states.append(states)
    sys.stdout.write("\b\b\b\b100%\n")
    return all_states


def plot_states(states):
    import pylab

    numChannels = len(states[0].X)
    numSamples = len(states)

    sampling=[state.t for state in states]
    y=numpy.array([state.X for state in states])
    maxy=numpy.ceil(y.max())

    for c in range(numChannels):
        pylab.plot(sampling, numpy.array(y[:,c]) + c*maxy)




def getLayerwiseStates(resp,layer_depth=[3,3,3],layer_base=None,**kwds):
    ''' ditto'''
    if not resp.__dict__.has_key('layers'):
        raise Exception('getLayerIdx not done')

    states = response2states(resp,**kwds)

    #get grid
    if layer_base==None:
        #assume square
        nx = numpy.sqrt(float(numpy.sum(resp.layeridx==0))/float(layer_depth[0]))
        if int(numpy.floor(nx)) != int(nx):
            raise Exception('not square. provide layer_base')
        layer_base = [nx,nx]


    idx_lst = []
    layer_lst = []
    for lay in range(len(resp.layers)):
        idx_lst.append(numpy.where(resp.layeridx==lay)[0])
        layer_lst.append(numpy.ones(layer_depth[lay])*lay)

    layer_lst = numpy.concatenate(layer_lst[::-1])

    for s in states[0]:
        X = []

        for lay in range(len(resp.layers)):
            X.append(s.X[idx_lst[lay]].reshape([layer_base[0],layer_base[1],layer_depth[lay]]))
    
        s.X = numpy.concatenate(X[::-1],axis=2)
        s.layeridx = layer_lst
        s.layers =resp.layers
        
    return states[0]



def plotLayerwiseStates(states,idx=None):

    from utils.mscfuncs import imagesc
    import pylab

    if idx ==None:
        idx = range(len(states))

    mx = 0.
    for i in idx:
        mx = numpy.max([states[i].X.max(), mx]);

    handle_lst = []
    for i in idx:
        handle_lst.append(pylab.figure())

        for lay in range(states[0].X.shape[2]):
            ax = pylab.subplot(3,3,lay+1)
            imagesc(states[i].X[:,:,lay])
            pylab.title('layer %s (%d) (t=%1.3fs)' % (states[i].layers[states[i].layeridx[lay]],lay,states[i].t),fontsize=8)
            pylab.clim([0,mx])
            pylab.setp(ax,'xticklabels',[])
            pylab.setp(ax,'yticklabels',[])

    return handle_lst




if __name__ == '__main__':
    import pylab


    if 0:

        Tsim=1.0
        freq=20
        
        resp = Response(Tsim)
        resp.appendChannel(Channel(data=numpy.random.uniform(0,Tsim, freq*Tsim)))
        resp.appendChannel(Channel(data=numpy.random.uniform(0,Tsim, freq*Tsim)))
        resp.appendChannel(Channel(data=numpy.random.uniform(0,Tsim, freq*Tsim)))
        resp.appendChannel(Channel(data=numpy.random.uniform(0,Tsim, freq*Tsim)))
        resp.appendChannel(Channel(data=numpy.random.uniform(0,Tsim, freq*Tsim)))
        
        sample=numpy.arange(0, Tsim, 0.001)

        #statesexp = response2states(resp, sample, spikes2exp)    
        #pylab.figure()
        #plot_states(statesexp)

        print "response2states spikes2exp"
        statesexp = response2states(resp, sample, spikes2exp)
        print "response2states spikes2alpha"
        statesalpha = response2states(resp, sample, spikes2alpha)
        print "response2states spikes2count"
        statescount = response2states(resp, sample, spikes2count)

        pylab.figure()
        pylab.subplot(311)
        plot_states(statesexp)
        pylab.subplot(312)
        plot_states(statesalpha)
        pylab.subplot(313)
        plot_states(statescount)
        pylab.savefig('states')

        #pylab.show()

    if 1:
        if 1:
            import dataformats
            #fname = '/home/malte/saves/pysim/sim_WInscaleNoRecurrent_ephysBackgroundMovie/WInscaleNoRecurrent_ephysBackgroundMovie.231'
            fname= '/home/malte/saves/pysim/sim_WscaleAndLRW_seed1434103_getStandardMovieStm_62x62_HC1/WscaleAndLRW_seed1434103_getStandardMovieStm_62x62_HC1.199'
            rsp = dataformats.Response()
            rsp.load(fname,'/Response')

            states = getLayerwiseStates(rsp,sampling=numpy.linspace(0,10.,50))
        

        plotLayerwiseStates(states)
        pylab.show()
