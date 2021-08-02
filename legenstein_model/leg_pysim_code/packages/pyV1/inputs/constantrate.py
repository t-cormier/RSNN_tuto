from inputgenerator import *
from spikes2rate import *
from rate2spikes import *

import numpy
import scipy
import pyV1.utils
from scipy.interpolate.interpolate import interp1d


class ConstantRate(RateGenerator):
    '''genarates poisson spike trains with a constant rate r(t) = f'''

    def __init__(self, **kwds):
        '''  nChannels ... number of spike trains (channels)
  Tstim     ... length of spike trains
  dt        ... time step for rate calculation
  f         ... rates of each individual spiking channels
'''        
        super(ConstantRate, self).__init__(**kwds)
        self.f = kwds.get('f', 80)               # maximal rate of a single channel
        
               
    def __str__(self):
        desc = '''  CONSTANT RATE
  nChannels  : %s
  Tstim      : %s
  f          : %s
  dt         : %s
''' % (self.nChannels, self.Tstim, self.f, self.dt)
        return desc
    
    
    def calcRate(self):
        t=numpy.array([0, self.Tstim])
        r=numpy.array([self.f, self.f])
        
        tt=numpy.arange(0, self.Tstim, self.dt)                
        r=interp1d(t,r,'linear')    # nearest not implemented yet
        
        return r(tt)
        

    
    def plot(self, stim, Tseg=0):
        import pylab
        import matplotlib
        matplotlib.rc('text', usetex=True)
        pylab.figure()
        pylab.subplot(2,1,1)
        
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
        pylab.legend(('r(t)', r'$r_{measured}$ ($\Delta=%s ms$)' % str(binwidth*1000)))


        pylab.subplot(2,1,2) # cla reset; hold on;
        
        for j in range(self.nChannels):
            st=stim.channel[j].data
            for spike in list(st):
                pylab.plot(numpy.array([spike, spike]), (numpy.array([[-0.3],[0.3]])+j+1), color='k')

        pylab.setp(pylab.gca(), xlim=[0, self.Tstim], ylim=[0.5, self.nChannels+0.5], yticks=numpy.arange(self.nChannels)+1)
        pylab.xlabel('time [s]')
        pylab.ylabel('channel')
        pylab.title('spike trains')#, fontweight='bold')

        

if __name__ == '__main__':
    import pylab
    rrate = ConstantRate(Tstim=1.0, nChannels=3)
    rrate2 = ConstantRate(Tstim=1.0, nChannels=3)
    stim = rrate.generate()
    
    rrate.plot(stim)
    pylab.show()
    
    