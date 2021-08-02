'''


    Klaus Schuch   schuch@igi.tugraz.at
    
    March 2007
'''

from inputgenerator import *
from spikes2rate import *
from rate2spikes import *

import numpy
import scipy

import pyV1.utils
from scipy.interpolate.interpolate import interp1d

class RandomRate(RateGenerator):
    '''genarates spike trains with a rate modulated by r(t) which is drawn randomly'''

    def __init__(self, **kwds):
        '''  nChannels ... number of spike trains (channels)
  Tstim     ... length of spike trains
  dt        ... time step for rate calculation
  nRates    ... number of different rates that can be chosen (equal spaced in the interval [0,fmax]
  binwidth  ... in each interval of length binwidth a new random rate is drawn
  fmax      ... maximal rate of a single channel
'''
        super(RandomRate, self).__init__(**kwds)
        self.nRates = float(kwds.get('nRates', numpy.inf))
        self.binwidth = kwds.get('binwidth', 30e-3)
        self.fmax = kwds.get('fmax', 80)
        
               
    def __str__(self):
        desc = '''  RANDOM RATE
  nChannels  : %s
  Tstim      : %s
  nRates     : %s
  binwidth   : %s
  fmax       : %s
  dt         : %s
''' % (self.nChannels, self.Tstim, self.nRates, self.binwidth, self.fmax, self.dt)
        return desc
    
    
    def calcRate(self):
        tt = numpy.arange(0, self.Tstim, self.dt)
        t = numpy.arange((-self.binwidth/2), self.Tstim+self.binwidth, self.binwidth)
                
        if self.nRates == numpy.inf:
            r = numpy.random.uniform(0, self.fmax, len(t))
        elif self.nRates > 1:
            r = numpy.ceil(random.uniform(0, self.nRates, len(t)))/(self.nRates)*self.fmax
        else:
            r = numpy.ones(len(t))*self.fmax
        
        r = interp1d(t, r, 'linear')    # nearest not implemented yet
        return r(tt)
        
    
    def plot(self, stim=None, Tseg=0):
        import pylab
        import matplotlib

        if stim is None:
            stim=self.generate()

        matplotlib.rc('text', usetex=True)
        pylab.figure()
        pylab.subplot(3,1,1)
        
        binwidth=30e-3 #self.binwidth
        tr=numpy.arange(0, stim.Tsim, stim.dt)
        (y, ty) = spikes2rate(utils.flatten([c.data for c in stim.channel]), binwidth)
        
        ty=ty.compress((y != numpy.nan).flat)
        y=y.compress((y != numpy.nan).flat)

        pylab.plot(tr, stim.r, ty, y/self.nChannels, 'r--')
        pylab.axis('tight')
        pylab.setp(pylab.gca(), xlim=[0, self.Tstim])   
        pylab.xlabel('time [s]')
        pylab.ylabel('rate per spike train [Hz]')

        pylab.title('rates')
        pylab.legend(('r(t)', r'$r_{measured}$ ($\Delta=%s~ms$)' % str(binwidth*1000)))


        pylab.subplot(3,1,2) # cla reset; hold on;
        
        for j in range(self.nChannels):
            st=stim.channel[j].data
            for spike in list(st):
                pylab.plot(numpy.array([spike, spike]), (numpy.array([[-0.3],[0.3]])+j+1), color='k')

        pylab.setp(pylab.gca(), xlim=[0, self.Tstim], ylim=[0.5, self.nChannels+0.5], yticks=numpy.arange(self.nChannels)+1)
        pylab.xlabel('time [s]')
        pylab.ylabel('channel')
        pylab.title('spike trains')#, fontweight='bold')


        pylab.subplot(3,1,3)
        if Tseg > 0:
            r=self.rand_rate(Tseg)
            cr=scipy.correlate(r-numpy.mean(r), r-numpy.mean(r), mode='full')
        else:
            Tseg=self.Tstim
            cr=scipy.correlate(stim.r-numpy.mean(stim.r), stim.r-numpy.mean(stim.r), mode='full')
                        
        cr=cr/max(cr)
        tr=(numpy.arange(len(cr))-len(cr)/2)*self.dt
        
        cs=scipy.correlate(y-y.mean(), y-y.mean(), mode='full')
        cs=cs/cs.max()
        
        ts=(numpy.arange(len(cs))-len(cs)/2)*self.dt
                
        pylab.plot(tr,cr,ts,cs,'r--')        
        pylab.xlabel('lag [s]')
        pylab.ylabel('correlation coeff')
        mm=max(abs(min(min(tr),min(ts))), abs(max(max(tr),max(ts))))
        
        pylab.setp(pylab.gca(), xlim=[-mm, mm], ylim=[min(cs), 1])
        pylab.title('auto-correlation', fontweight='bold')
        pylab.legend(('r(t)', r'$r_{measured}$ ($\Delta=%s~ms$)' % str(binwidth*1000)))
        

if __name__ == '__main__':
    import pylab
    rrate = RandomRate(Tstim=1.0, nChannels=3)

#    stim = rrate.generate()
#    rrate.plot(stim)

    rrate.plot()
    
    pylab.show()
    
    