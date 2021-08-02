from inputgenerator import *

import numpy
import scipy.io
import scipy.weave as wv
import pyV1.utils.default as default


class EphysRandomRate(MultdimensionalRateGenerator):
    '''genarates spike trains with the distribution of ephys data'''

    def __init__(self, **kwds):
        '''  nChannels ... number of spike trains (channels)
        Tstim     ... length of spike trains
        dt        ... time step for rate calculation
        stmtype   ... spont /movie
        area      ... V1 or LGN data
        rorc ... rate of rate change (Poisson events) 
        '''
        super(EphysRandomRate, self).__init__(**kwds)

        self.stmtype = kwds.get('stmtype','spont')
        self.area = kwds.get('area','LGN')
        self.rorc = kwds.get('rorc',0.)

        self.loadpath = default.dirs['ephys']
        
               
    def __str__(self):
        desc = '''  EphysRandomRate
        nChannels  : %s
        Tstim      : %s
        stmtype    : %s
        area       : %s
        dt         : %s
        ''' % (self.nChannels, self.Tstim, self.stmtype,self.area, self.dt)
        return desc
    
    
    def calcRate(self):

        tt = numpy.arange(0, self.Tstim, self.dt)
        
        ratedists = scipy.io.loadmat(self.loadpath + 'ratedists')['ratedists']
                
        p = ratedists.__dict__[self.stmtype].__dict__[self.area].__dict__['p']
        rate = ratedists.__dict__[self.stmtype].__dict__[self.area].__dict__['x']

        #partitition
        ppart = numpy.cumsum(p)

        #one for rate changes
        shape = [self.nChannels, len(tt)]
        Rmat = (numpy.random.rand(*shape) < self.dt*self.rorc).astype(numpy.float)

        Rmat[:,0] = 1. #first changes

        #draw rates
        nrates = int(Rmat.sum())

        r = numpy.random.rand(nrates)      
        
        code = """
        int i,t,l;
        int s = 0;
        double actfr;

        int ni = (int) shape[0];
        int nt = (int) shape[1];
        
        for (i=0;i< ni;i++){
           actfr = 0.;
           for (t=0;t< nt;t++){
              if (Rmat[i*nt + t]) {
                 for (l=0;l<(int) nrates;l++){
                     if ((double) ppart[l] >= (double) r[s]){
                        actfr= rate[l];
                        break;
                     }
                 }
                 s++;
               }
            Rmat[i*nt + t] = actfr;
            }
        }
        """
        shape = Rmat.shape
        wv.inline(code, ['shape','rate','Rmat','ppart','r','nrates'])

        return Rmat

    
if __name__ == '__main__':
    import pylab
    rrate = EphysRandomRate(Tstim=2.0, nChannels=100,dt=0.001,rorc=0.5,area='V1',stmtype='movie')
    
    stim = rrate.generate()

    stim.plot()
    pylab.show()
