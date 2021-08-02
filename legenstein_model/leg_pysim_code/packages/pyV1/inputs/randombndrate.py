import numpy

from scipy.interpolate.interpolate import interp1d
from randomrate import *


class RandomBndRate(RandomRate):
    '''genarates spike trains with a rate modulated by r(t) which is drawn randomly'''

    def __init__(self, **kwds):
        '''  nChannels ... number of spike trains (channels)
  Tstim     ... length of spike trains
  dt        ... time step for rate calculation
  nRates    ... number of different rates that can be chosen (equal spaced in the interval [0,fmax]
  binwidth  ... in each interval of length binwidth a new random rate is drawn
  fmax      ... maximal rate of a single channel
  fmin      ... minimal rate of a single channel
'''
        super(RandomBndRate, self).__init__(**kwds)        
        self.fmin = kwds.get('fmin', 0.0)                # minimal rate of a single channel
        
               
    def __str__(self):
        desc = '''  RANDOM RATE
  nChannels  : %s
  Tstim      : %s
  nRates     : %s
  binwidth   : %s
  fmin       : %s
  fmax       : %s
  dt         : %s
''' % (self.nChannels, self.Tstim, self.nRates, self.binwidth, self.fmin, self.fmax, self.dt)
        return desc
        

    def calcRate(self):
        tt=numpy.arange(0, self.Tstim, self.dt)
        t=numpy.arange((-self.binwidth/2), self.Tstim+self.binwidth, self.binwidth)
                
        if self.nRates == numpy.inf:
            r=numpy.random.uniform(0, (self.fmax - self.fmin), len(t)) + self.fmin
        elif self.nRates > 1:
            r=numpy.floor(numpy.random.uniform(0, (self.nRates+1), len(t)))/(self.nRates)*(self.fmax-self.fmin) + self.fmin
        
        r=interp1d(t,r,'linear')    # nearest not implemented yet


        
        return r(tt)
            

if __name__ == '__main__':
    import pylab
    rrate = RandomBndRate(Tstim=1.0, fmin=20.0, fmax=40.0)
    rrate2 = RandomBndRate(Tstim=1.0, fmin=40.0, fmax=80.0)
    
#    stim = rrate.generate()
#    stim2 = rrate2.generate()
#    rrate.plot(stim)
#    rrate2.plot(stim2)
    
    rrate.plot()
    rrate2.plot()

    pylab.show()
