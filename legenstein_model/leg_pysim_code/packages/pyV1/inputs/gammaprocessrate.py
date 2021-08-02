'''
Implemented a switching gamma renewal process according to (Gazere et al., 1998)
A phenomenological model of visual evoked spike trains in cat geniculate nonlagged X-cells.
Visual Neuroscience (1998) 15, 1157-1174

    Klaus Schuch   schuch@igi.tugraz.at
    
    March 2007
'''

from inputgenerator import *
import numpy

from pyV1.utils.logger import logger
import scipy.weave as wv

from time import time


def rate2spikes_sgp_old(u, dt, Tmax, reg, regsw):
    '''  Implemented a switching gamma renewal process according to (Gazere et al., 1998)
  A phenomenological model of visual evoked spike trains in cat geniculate nonlagged X-cells.
  Visual Neuroscience (1998) 15, 1157-1174
'''
    u = numpy.atleast_1d(u)
    reg = numpy.atleast_1d(reg)
    regsw = numpy.atleast_1d(regsw)

    # stimulus time if every time segment of length dt is scaled by the frequency    
    TmaxUniform = sum(u*dt); 
    nbin = len(u)

    # assign regularity values to each time segment dt
    uu = numpy.tile(u,(len(reg),1)).transpose()
    ridx = numpy.sum(uu >= regsw,1)-1
    r = reg[ridx]

    # generate stationary gamma processes for each r and constant frequency of 1 Hz
    
    # pre calculate spike times and inter spike intervals for each regularity and constant rate 1 HZ 
    ISI = []
    SPT = []
    for i in range(len(reg)):
        t0 = 0.0
        lambd = 1.0/reg[i] # so that f = lambda*r = 1
        
        # choose 1st ISI according to equ (8)
        k = numpy.ceil(reg[i]*numpy.random.uniform(0,1))
        ISI.append(numpy.random.gamma(k, lambd, 1))
        
        # choose later ISI according to equ (4)
        while t0 < TmaxUniform:
            isi = numpy.random.gamma(reg[i], lambd, (1e3))
            ISI[i]= numpy.r_[ISI[i], isi]
            t0 += numpy.sum(isi) #sum(ISI[i]) MJR
        
        tsum = numpy.cumsum(ISI[i])
        
        ISI[i] = ISI[i].compress((tsum < TmaxUniform).flat)
        SPT.append(tsum.compress((tsum < TmaxUniform).flat))

    # project the spike train from each regularity r spike train to the output spike train (with appropriate scaling)
    tiuniform = numpy.cumsum(dt*u)
    tiuniform = numpy.r_[0, tiuniform[:-1]]
    
    ti = numpy.arange(0, Tmax, dt)
    spike_train = []
    
    for i in range(len(reg)):
        nSpikes = len(SPT[i])
    
        # check in which time bin each spike is located
        m = numpy.kron(numpy.ones((nbin, 1)), numpy.atleast_2d(SPT[i].T)).T

        if nSpikes > 0:
            b = numpy.kron(numpy.ones((nSpikes, 1)), numpy.atleast_2d(tiuniform))
            m = m - b
        
        tbinidx = numpy.sum(m > 0, 1) - 1
        
        ri = numpy.array(ridx).take(tbinidx)
        
        # keep only spikes that are located in time bins with correct regularity index
        di = numpy.where(ri == i)[0]
        
        SPT[i] = SPT[i][di]
        tbinidx = tbinidx[di]
        
        # rescale time axis
        spt = (SPT[i] - tiuniform[tbinidx]) / u[tbinidx] + ti[tbinidx]
        spike_train = numpy.r_[spike_train, spt]  

    spike_train.sort()

    return spike_train



def rate2spikes_sgp(u, dt, Tmax, reg, regsw):
    '''  Implemented a switching gamma renewal process according to (Gazere et al., 1998)
  A phenomenological model of visual evoked spike trains in cat geniculate nonlagged X-cells.
  Visual Neuroscience (1998) 15, 1157-1174
'''
    u = numpy.atleast_1d(u)
    reg = numpy.atleast_1d(reg)
    regsw = numpy.atleast_1d(regsw)

    # stimulus time if every time segment of length dt is scaled by the frequency    
    TmaxUniform = sum(u*dt); 
    nbin = len(u)

    # assign regularity values to each time segment dt
    uu = numpy.tile(u,(len(reg),1)).transpose()
    ridx = numpy.sum(uu >= regsw,1)-1
    r = reg[ridx]

    # generate stationary gamma processes for each r and constant frequency of 1 Hz
    
    # pre calculate spike times and inter spike intervals for each regularity and constant rate 1 HZ 
    ISI = []
    SPT = []
    for i in range(len(reg)):
        t0 = 0.0
        lambd = 1.0/reg[i] # so that f = lambda*r = 1
        
        # choose 1st ISI according to equ (8)
        k = numpy.ceil(reg[i]*numpy.random.uniform(0,1))
        ISI.append(numpy.random.gamma(k, lambd, 1))
        
        # choose later ISI according to equ (4)
        while t0 < TmaxUniform:
            isi = numpy.random.gamma(reg[i], lambd, (1e3))
            ISI[i]= numpy.r_[ISI[i], isi]
            t0 += numpy.sum(isi) #sum(ISI[i]) MJR
        
        tsum = numpy.cumsum(ISI[i])
        
        ISI[i] = ISI[i].compress((tsum < TmaxUniform).flat)
        SPT.append(tsum.compress((tsum < TmaxUniform).flat))

    # project the spike train from each regularity r spike train to the output spike train (with appropriate scaling)

    code ="""
    int i, j;

    for (i=0, j=0; i<nbin; i++) {
        if  (tiuniform[i]>spk[j]) {
            tbinidx[j] = i-1;
            j++;
            i--;
            if (j>=nSpikes) {break;}  
        }

        if  ((i==nbin-1) && (tiuniform[i]<=spk[j])) {
           tbinidx[j] = i;
           j++;
           i--;
           if (j>=nSpikes) {break;}  
        }
    }
    
    """

    tiuniform = numpy.cumsum(dt*u)
    tiuniform = numpy.r_[0, tiuniform[:-1]]
    
    ti = numpy.arange(0, Tmax, dt)
    spike_train = []

    OLD = False

    for i in range(len(reg)):
        nSpikes = len(SPT[i])
    
        # check in which time bin each spike is located

        if OLD:
            m = numpy.kron(numpy.ones((nbin, 1)), numpy.atleast_2d(SPT[i].T)).T
        
            if nSpikes > 0:
                b = numpy.kron(numpy.ones((nSpikes, 1)), numpy.atleast_2d(tiuniform))
                m = m - b
        
            tbinidx2 = numpy.sum(m > 0, 1) - 1

        tbinidx = numpy.zeros(nSpikes).astype(numpy.int);
        spk = SPT[i]
        wv.inline(code,['tbinidx','nSpikes','spk','nbin','tiuniform'])


        if OLD:
            print numpy.all(tbinidx2==tbinidx)

            if not numpy.all(tbinidx2==tbinidx):                
                import pdb
                pdb.set_trace()
         
        ri = ridx.take(tbinidx)

        # keep only spikes that are located in time bins with correct regularity index
        di = numpy.where(ri == i)[0]
        
        tbinidx_di = tbinidx[di]
        
        # rescale time axis
        spt = (SPT[i][di].copy() - tiuniform[tbinidx_di]) / u[tbinidx_di] + ti[tbinidx_di]
        spike_train = numpy.r_[spike_train, spt]  

    spike_train.sort()

    return spike_train


class GammaProcessRate(InputGenerator):
    '''  GAMMA_PROCESS_RATE
    generates gamma process spike trains for channel i with rate r_i(t) = f_i'''


    def __init__(self, **kwds):
        '''  nChannels         ... number of spike trains (channels)
  Tstim             ... length of stimulus (-1 ... specify the length in the call to generate)
  f                 ... rates of each individual spiking channels
  gamma_freq_switch ... Frequency switches for gamma_r
  gamma_r           ... Regularity parameter at different gamma_freq_switch
  cluster           ... if not None the spike generation is computed distributed 
  '''
        self.Tstim = float(kwds.get('Tstim', -1.0))
        self.f = kwds.get('f', numpy.array([[20.0, 65.0, 30, 60], [30, 62, 40, 60]]))
        self.gamma_freq_switch = kwds.get('gamma_freq_switch', numpy.array([0.0, 60.0]))
        self.gamma_r = kwds.get('gamma_r', numpy.array([1, 5]))
        self.cluster = kwds.get('cluster', None)
        self.mincpe = 50        # minimum number of channels per engine (task)


    def __str__(self):
        desc = '''  GAMMA_PROCESS_RATE
  Tstim            : %s
  f                : %s
  gamm_freq_switch : %s
  gamm_r           : %s
''' % (self.Tstim, self.f, self.gamma_freq_switch, self.gamma_r)
        return desc


    def generate(self, argTstim=-1):
        '''generates a Stimulus object'''

        if self.Tstim > -1:
            Tstim=self.Tstim
        else:
            Tstim=argTstim

        if Tstim <= 0.0:
            error('length of stimulus undefined!')
        
        # channel number in first dim and time in second dim of f
        if numpy.isscalar(self.f):
            nTimeSlots = 1
            nChannels = 1
        elif self.f.ndim < 2:
            nTimeSlots = 1
            nChannels = self.f.shape[0]
        else:
            (nChannels, nTimeSlots) = self.f.shape
                
        dt = Tstim/nTimeSlots
              
        reg = self.gamma_r
        regsw = self.gamma_freq_switch

        stimulus = Stimulus(self.Tstim)
        
        # get Spikes for each channel seperately (different rate)
        if self.cluster is None:
            for i in range(nChannels):
                r = self.f[i]
                spikes = rate2spikes_sgp(r, dt, Tstim, numpy.atleast_1d(reg), numpy.atleast_1d(regsw))
                stimulus.appendChannel(Channel(spikes))
        else:
            import ipython1.kernel.api as kernel

            rc = self.cluster.getRemoteControllerClient()
            tc = self.cluster.getTaskControllerClient()
            nEngines =len(rc.getIDs())
            tids=[]

            rc.resetAll()
            rc.executeAll('import inputs.gammaprocessrate as gpr\n')
            #rc.pushAll(dt=dt, Tstim=Tstim, reg=reg, regsw=regsw)
            
            cmd = '\nspikes=[]\nfor r in rlist:\n    spikes.append(gpr.rate2spikes_sgp(r, dt, Tstim, reg, regsw))\n'

            nTasks=nEngines
            cpe=int(numpy.ceil(float(nChannels)/nTasks))

            if cpe < self.mincpe:
                cpe=self.mincpe
                nTasks=int(numpy.ceil(float(nChannels)/cpe))

            for i in range(nTasks):
                endidx = i*cpe+cpe
                if endidx < nChannels:
                    r= self.f[i*cpe:endidx]
                else:
                    r= self.f[i*cpe:]
                setupNS = dict(rlist=r,dt=dt, Tstim=Tstim, reg=reg, regsw=regsw)
                tids.append(tc.run(kernel.Task(cmd, resultNames=['spikes'], setupNS=setupNS)))

            logger.info(' gammaprocessrate(): %d tasks started' % len(tids))
            tc.barrier(tids)
            for tid in range(len(tids)):
                res = tc.getTaskResult(tids[tid])
                for sp in res.results['spikes']:
                    stimulus.appendChannel(Channel(sp))

        stimulus.dt = dt

        return stimulus


 
if __name__ == '__main__':
    import pylab
    pylab.figure()

    f = numpy.array([49.9, 49.9, 49.9, 49.9])
    f = numpy.kron(numpy.ones((30, 1)), numpy.atleast_2d(f))
    gammaprocess=GammaProcessRate(Tstim=1.0, gamma_freq_switch=[0, 50], f=f, gamma_r=[1, 5])
    stim1=gammaprocess.generate()
    pylab.subplot(3,1,1)
    stim1.plot()

    f = numpy.array([50.0, 50.0, 50, 50])
    f = numpy.kron(numpy.ones((30, 1)), numpy.atleast_2d(f))
    gammaprocess=GammaProcessRate(Tstim=1.0, gamma_freq_switch=[0, 50], f=f, gamma_r=[1, 5])
    stim1=gammaprocess.generate()
    pylab.subplot(3,1,2)
    stim1.plot()

    f = numpy.array([49.9, 50.0, 49.9, 50])
    f = numpy.kron(numpy.ones((30, 1)), numpy.atleast_2d(f))
    gammaprocess=GammaProcessRate(Tstim=1.0, gamma_freq_switch=[0, 50], f=f, gamma_r=[1, 5])
    stim1=gammaprocess.generate()
    pylab.subplot(3,1,3)
    pylab.xlabel('time (s)')
    stim1.plot()
    
    pylab.show()
