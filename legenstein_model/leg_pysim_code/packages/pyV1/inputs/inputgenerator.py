'''
Basic InputGenerators

    Klaus Schuch   schuch@igi.tugraz.at
    
    March 2007
'''

from pyV1.dataformats import Stimulus, Channel
from spikes2rate import spikes2rate
from rate2spikes import rate2spikes
import pyV1.utils.hdf5pickle as hdf5
import tables
import numpy


class InputGenerator(object):
    def generate(self):
        pass

    def __str__(self):
        desc = "  INPUT_GENERATOR\n"
        return desc




class RateStimulus(Stimulus):
    def __init__(self, Tsim=0.5, r=None, dt=1e-3):
        super(RateStimulus, self).__init__(Tsim=Tsim)
        if r is None:
            self.r = []
        else:
            self.r = r
        self.dt = dt

    def __str__(self):
        desc = """  RandomRateStimulus
  channel : [1x%s struct]
  Tsim    : %s
  r       : %s\n""" % (len(self.channel), self.Tsim, self.r)
        return desc    




class RateGenerator(InputGenerator):    
    '''genarates spike trains with a given rate, the method calcRate() has to be overloaded'''

    def __init__(self, **kwds):
        '''  nChannels ... number of spike trains (channels)
  Tstim     ... length of spike trains
  dt        ... time step for rate calculation
'''
        super(RateGenerator, self).__init__(**kwds)
        self.nChannels = kwds.get('nChannels', 4)
        self.Tstim = float(kwds.get('Tstim', 1.0))        
        self.dt = kwds.get('dt', 1e-3)

    def calcRate(self):
        return []
    
    def generate(self):
        r = self.calcRate()
        spikes=rate2spikes(r, self.dt, self.nChannels, self.Tstim)      
        stimulus = RateStimulus(self.Tstim, r, self.dt);        
        for i in range(self.nChannels):
            stimulus.channel.append(Channel(spikes[i]))
                        
        return stimulus





class MultdimensionalRateGenerator(InputGenerator):    
    '''genarates spike trains with a given rate for each channel, the
    method calcRate() has to be overloaded'''

    def __init__(self, **kwds):
        '''  nChannels ... number of spike trains (channels)
  Tstim     ... length of spike trains
  dt        ... time step for rate calculation
'''
        super(MultdimensionalRateGenerator, self).__init__(**kwds)
        self.nChannels = kwds.get('nChannels', 4)
        self.Tstim = float(kwds.get('Tstim', 1.0))        
        self.dt = kwds.get('dt', 1e-3)

    def calcRate(self):
        return []
    
    def generate(self):
        rmat = self.calcRate()
        stimulus = RateStimulus(self.Tstim, rmat, self.dt);        
        for r in rmat:
            spikes=rate2spikes(r, self.dt, 1, self.Tstim)      
            stimulus.channel.append(Channel(spikes[0]))

        return stimulus





class CombinedStimulus(Stimulus):
    def __init__(self, Tsim=0.5):
        super(CombinedStimulus, self).__init__(Tsim=Tsim)
        self.stim_list = [] 
        #self.save_attrs.append('stim_list')


#!!!!!! # MJR:appending to save_attrs not possible: too big and will be saved twice!!!
# maybe not save the whole object in stim_lst but the channel ID or so (or ref anyway?)

    def __str__(self):
        desc = '''  CombinedStimulus
  channel        : [1x%s struct]
  Tsim          : %s
  stim_list     : %s\n''' % (len(self.channel), self.Tsim, str(self.stim_list))
        return desc
    
    def append(self, stim):
        self.stim_list.append(stim)
        
        super(CombinedStimulus, self).append(stim)





class CombinedInputGenerator(object):
    ''' COMBINED INPUT GENERATOR
    wrapps a list of INPUTGENERATORS
    '''
    def __init__(self, genlist=None):
        '''  genlist ... list of InputGenerators
'''
        if genlist is None:
            self.gelist=[]
            self.Tstim = None
        else:
            self.genlist=genlist
            self.Tstim=max([gen.Tstim for gen in genlist])
        

    def save(self,fname,where='combinedInput'):

        cl_lst = []
        dd = []
        h = tables.openFile(fname,'w')
        if where!=None:
            h.createGroup('/',where)
        h.close()
        
        for i in range(len(self.genlist)):
            grp = '/' + where + '/d' + str(i)

            hdf5.dumpDict(self.genlist[i].__dict__,fname,grp)
            cl_lst.append(self.genlist[i].__class__)


        grp = where + '/c' 
        hdf5.dumpDict({'_class':cl_lst},fname,grp)


    def load(self,fname,where='combinedInput'):


        cl = hdf5.loadDict(fname,'/' + where + '/c')['_class']

        h = tables.openFile(fname)
        grp = h.getNode('/'+where)
        self.genlist =numpy.zeros(len(grp._v_groups)-1).tolist()
        
        grps = dict(grp._v_groups).keys()
        h.close()

        for g in grps:
            if g=='c':
                continue
            d = hdf5.loadDict(fname,'/' + where+'/'+g)
            idx = int(g[1:])
            self.genlist[idx] = cl[idx]()
            self.genlist[idx].__dict__.update(d)

        self.Tstim=max([gen.Tstim for gen in self.genlist])
        

    
    def generate(self):
        ''' generates the Input of all the Input-Generators'''
        stim=CombinedStimulus(Tsim=self.Tstim)
        
        for i in range(len(self.genlist)):
            s=self.genlist[i].generate()
            
#            if i==0:
#                stim=s
#            else:
#                stim.append(s)

            stim.append(s)

#        for gen in self.genlist:
#            stim.append(gen.generate())
    
        return stim
    

    def __str__(self):
        desc = "  COMBINED_INPUT_GENERATOR:\n"
        for gen in self.genlist:
            desc += gen.__str__()
            
        return desc



if __name__ =='__main__':
    
    import paramsearch.experiments as e
    import inputs.ephysrate

    Tsim = 1.
    dt = 0.01
    pars = e.getSmallPars()
    pars.lgndic['_class'] ='lgn.LGNEphysGammaBackground'
    pars.lgndic['rorc'] = 0.5 #sec
    pars.lgndic['stmtype'] = 'movie' #sec
    pars.lgndic['area'] = 'LGN' #sec
    stim = []
    lgn = pars.getLGN()

    indims = lgn.dims
    #rRate =inputs.constantrate.ConstantRate(nChannels=(indims[0]*indims[1]), Tstim=Tsim, dt=dt, f=9.9)
    rRate =inputs.ephysrate.EphysRandomRate(nChannels=(indims[0]*indims[1]), Tstim=Tsim, dt=dt, area='V1',stmtype='spont',rorc=0.5)

    lgn.processStimulus(stim,Tsim,dt)

    input = CombinedInputGenerator([ lgn, rRate])


    input.save('test14.h')
    input2 = CombinedInputGenerator()
    input2.load('test14.h')
