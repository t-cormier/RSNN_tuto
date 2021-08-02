import numpy

from scipy.interpolate.interpolate import interp1d
from randombndrate import *


class RandomModRate(RandomBndRate):
    '''genarates spike trains with a rate modulated by a sinusoidal rate r(t) whose frequency is drawn randomly'''

    def __init__(self, **kwds):
        '''  nChannels ... number of spike trains (channels)
  Tstim     ... length of spike trains
  dt        ... time step for rate calculation
  nRates    ... number of different rates that can be chosen (equal spaced in the interval [0,fmax]
  binwidth  ... in each interval of length binwidth a new random rate is drawn
  freq      ... frequency od the underlying signal
  fmax      ... maximal rate of a single channel
  fmin      ... minimal rate of a single channel
'''
        super(RandomModRate, self).__init__(**kwds)        
        self.freq = kwds.get('freq', 40.0)                # minimal rate of a single channel
        
               
    def __str__(self):
        desc = '''  RANDOM MOD RATE
  nChannels  : %s
  Tstim      : %s
  nRates     : %s
  binwidth   : %s
  freq       : %s
  fmin       : %s
  fmax       : %s
  dt         : %s
''' % (self.nChannels, self.Tstim, self.nRates, self.binwidth, self.freq, self.fmin, self.fmax, self.dt)
        return desc
        

    def calcRate(self):
        tt=numpy.arange(0, self.Tstim, self.dt)
        t=numpy.arange((-self.binwidth/2), self.Tstim+self.binwidth, self.binwidth)

        nSegment=numpy.round(self.Tstim/self.binwidth)
        ttseg=numpy.arange(0, self.binwidth-self.dt/2, self.dt)
        r=[]; t=[]        
        
        for nS in range(nSegment+1):
            if self.nRates == numpy.inf:
                f=numpy.random.uniform(0, (self.fmax-self.fmin)) + self.fmin
            else:
                f=numpy.floor(numpy.random.uniform(0, self.nRates+1))/self.nRates*(self.fmax-self.fmin) + self.fmin
            
            pr = numpy.sin(f*ttseg*2*numpy.pi)
            r = numpy.r_[r, pr]
            t = numpy.r_[t, (ttseg + nS*self.binwidth)]           
                
        r=interp1d(t, self.freq*r/2 + self.freq/2, 'linear')
        
        return r(tt)
    

if __name__ == '__main__':
    import pylab
    rrate = RandomModRate(Tstim=1.0, fmin=20.0, fmax=40.0)
    rrate2 = RandomModRate(Tstim=1.0, fmin=40.0, fmax=80.0)

    rrate.plot()
    rrate2.plot()

#    stim = rrate.generate()
#    stim2 = rrate2.generate()    
#    rrate.plot(stim)
#    rrate2.plot(stim2)
    
    pylab.show()
    
    