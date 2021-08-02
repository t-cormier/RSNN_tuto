from dataformats import *

import numpy
import utils
import matplotlib.mlab

#import timedtest


class PSD(object):
	def __init__(self, power=[], freq=[]):
		if power is None:
			power = []
		self.power = power
		
		if freq is None:
			freq = []
		self.freq = freq
	
	def plot(self, type='loglog', **kwargs):
		if type=='loglog':
			pylab.loglog(self.freq, self.power, **kwargs)
			
		elif type=='loglin':
			pylab.plot(self.freq, 10*numpy.log10(self.power), **kwargs)
			vmin, vmax = pylab.gca().viewLim.intervaly().get_bounds()
			intv = vmax-vmin
			logi = int(numpy.log10(intv))
			if logi==0: 
				logi=.1
			step = 10*logi
			ticks = numpy.arange(math.floor(vmin), math.ceil(vmax)+1, step)
			pylab.gca().set_yticks(ticks)

		pylab.xlabel('frequency [Hz]')
		pylab.title('Power spectral density')
		pylab.ylabel('power [db]')
		pylab.grid(True)
		
		
#@timedtest.timedtest
def createC(spiketimes, dt, tau=0.01, T=None):
    ''' computes the driving currents of LFP '''

    spiketimes = utils.flatten(spiketimes)			# TODO better
    spiketimes = numpy.atleast_1d(spiketimes)    
    
    if T is None:
        T=numpy.ceil(spiketimes.max())			# TODO better ;)
    
    D=numpy.zeros(T/dt)
    for spike in spiketimes:
        D[int(spike/dt)] = 1
        
    n = numpy.arange(0, numpy.ceil(5*tau/dt))
    h = numpy.exp(-n*dt/tau)
    C = numpy.convolve(D, h, mode='same')     
    
    return C

    
def lfpPSD(spiketimes, dt=1e-4, tau=0.01, T=None):
    ''' computes the PSD of the LFP given the spiketimes
    (see C. Bedard, H. Kroeger and A. Destexhe (2006)'''
    
    C = createC(spiketimes, dt, tau, T)
    
    (power, freq) = matplotlib.mlab.psd(C, NFFT=256, Fs=1/dt) 
    psd = PSD(power, freq)
    
    # apply 1/f filter
    filtered_power = numpy.zeros(len(power))
    filtered_power[0] = power[0]
    filtered_power[1:-1] = power[1:-1]*1/freq[1:-1]
   
    filtered_psd=PSD(filtered_power, freq)
    
    return (psd, filtered_psd)


def calcpsd(data, dt=1e-4):
    (power, freq) = matplotlib.mlab.psd(data, NFFT=2048, Fs=1/dt) 
    p = PSD(power, freq)
    return p


def calcISI(spiketimes):
    isi=numpy.atleast_1d()

    for spikelist in spiketimes:
        isi=numpy.r_[isi, numpy.diff(spikelist)]

    return isi


if __name__ == '__main__':
    import pylab

    dt=1e-4
    T=10.0
    tau=0.01
    nChannels=100
    freq=40.0
    
    spikes = [(numpy.random.uniform(0, T, int(freq*T))) for i in range(0, nChannels)]
    
    for s in spikes:
    	s.sort()
    	    
    C=createC(spikes, dt, tau, T)
    
    pylab.figure()
    pylab.plot(numpy.arange(0,T,dt), C)

    (psd, filtered_psd) = lfpPSD(spikes, dt, tau, T)
    
    pylab.figure()
    psd.plot()
    
    pylab.figure()
    filtered_psd.plot()    
    
    pylab.show()
    