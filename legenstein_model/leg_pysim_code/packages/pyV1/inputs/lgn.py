'''
Implementation of a simple Retina-LGN model
(a 'sequence of bitmaps to spikes converter')
 
The model for the LGN closely follows the standard design 
(Ferster, 1990; Dong and Atick, 1995, Troyer et al., 1998) 
and resembles a spatio-temporal filter bank with non-linearities.

The Retina converts time varying input signals into firing rates of LGN neurons. 
Each visual input to the retina evokes four different types of 
firing rate responses of the retina-LGN model corresponding to 
the two diffeerent types of LGN cells, i.e. of non-lagged or lagged cells, 
in combination with the two types of retinal cells, 
i.e. on-center or off-center cells. Lagged and non-lagged LGN cells have been
observed experimentally 
(Mastronarde, 1987; Humphrey and Weller, 1988a; Humphrey and Weller, 1988b). 
From these firing rates spike trains are generated as LGN output. 

This standard model was extended by two features.
Additional to the short transient high-frequency phasic discharge of LGN cells 
(in response to a small light spot) a longer lasting tonic discharge 
was implemented with temporal dynamics taken from CNRS (Gazeres et al., 1998).

Secondly, the difference in firing statistics for the initial phasic response 
and the tonic response was accounted for by implementing a switching gamma renewal process 
(Gazeres et al., 1998) for the spike generation mechanism.


    Klaus Schuch   schuch@igi.tugraz.at
    
    March 2007
'''

import gammaprocessrate as gammagen
import ephysrate

import scipy
import scipy.signal
import numpy
#import pylab

import types
import matplotlib.mlab

import os
import time

from utils.logger import logger
from utils import default


#-----------------------------------------------------------------------------

def concBitmap(s, ind):
    emptycol = numpy.zeros([s.shape[0], 2])

    V = emptycol
    for t in ind:
        V = numpy.hstack((V, s[:,:,t]))
        V = numpy.hstack((V, emptycol))

    return V


def DoG(dims, sig_inner, sig_outer, omega = 1.0, substractMean=True,separateKernels=False, horizon=3.5):
    '''  generates a Difference of Gaussians kernel with
    dims      ... dim[0] - x-size, dim[1] - y-size of the kernel
    sig_inner ... std deviation of the inner gaussian
    sig_outer ... std deviation of the outer gaussian
    omega     ... ratio of amplitudes A_outer/A_inner
    substractmean ... whether to substract the mean (default: True)
    '''

    dims = numpy.atleast_1d(dims)

    # odd number of dim
    r = (dims % 2)
    dimo = dims + (r==0)

    h = numpy.ceil(horizon*sig_outer)
    dimo[dimo>h] = h

    if sig_outer < sig_inner:
        sig_outer, sig_inner = sig_inner, sig_outer

    x = numpy.arange(numpy.ceil(-dimo[0]/2.0), numpy.ceil(dimo[0]/2.0))
    y = numpy.arange(numpy.ceil(-dimo[1]/2.0), numpy.ceil(dimo[1]/2.0))
    (X, Y) = numpy.meshgrid(x, y)

    Zinner = matplotlib.mlab.bivariate_normal(X, Y, sig_inner, sig_inner, 0.0, 0.0)[:dims[1],:dims[0]]
    Zouter = matplotlib.mlab.bivariate_normal(X, Y, sig_outer, sig_outer, 0.0, 0.0)[:dims[1],:dims[0]]
     
    if not separateKernels:
        out = Zinner - omega * Zouter

        if substractMean:
            out = out - out.mean()

        return out
    else:
        return (Zinner,omega * Zouter)
    

def LgnKernel(t, wc):
    '''  generates a temporal LGN -kernel
    t    ... array of times to evaluate the kernel
    wc   ... 'qian'
         ... 'dong'
         ... 'dongtonic'
         ... number
    '''
    if type(wc) is types.StringType:
        if wc=='qian':
            return LgnQianKernel(t)
        elif wc=='dong':
            return LgnDongKernel(t)
        elif wc=='dongtonic':
            return LgnDongTonicKernel(t)
        elif wc=='donglowtonic':
            return LgnDongLowTonicKernel(t)
        else:
            raise Exception('LGN temporal Kernel unspecified')
        
    out = t*(1 - numpy.pi*wc*t) * exp(-2*numpy.pi*wc*t)
    out = out/out.max()
    
    out = out - out.mean()

    return out


def LgnQianKernel(t):
    '''  generates a temporal LGN -kernel according to Qian
    t    ... array of times to evaluate the kernel
    '''
    tau = 0.016
    omega = 4*2*numpy.pi    # 4 Hz
    phi = 0.24

    out = t/tau**2*exp(-t/tau) * cos(omega*tau + phi)
    out = out/out.max()

    return out


def LgnDongKernel(t):
    '''  generates a temporal LGN -kernel according to Dong
    t    ... array of times to evaluate the kernel
    '''
    return LgnKernel(t, 5.5)


def LgnDongTonicKernel(t):
    '''  generates a tonic temporal LGN -kernel according to Dong
    t    ... array of times torem evaluate the kernel
    '''
    wwc = 5.5
    tonicA = 0.3
    tonicTau = 0.015
    
    out = t * (1 - numpy.pi*wwc*t) * numpy.exp(-2*numpy.pi*wwc*t)
    out = out - out.mean()
    
    outTonic = numpy.exp(-t/tonicTau)
    
    cs1 = out/numpy.max(numpy.cumsum(out))
    cs2 = tonicA*outTonic/numpy.max(numpy.cumsum(outTonic))
    
    return cs1 + cs2


def LgnDongLowTonicKernel(t,wwc=5.5,tonicA=0.05,tonicTau=0.015):
    '''  generates a tonic temporal LGN -kernel according to Dong
    t    ... array of times torem evaluate the kernel
    '''
    #wwc = 5.5
    #tonicA = 0.05
    #tonicTau = 0.015
    
    out = t * (1 - numpy.pi*wwc*t) * numpy.exp(-2*numpy.pi*wwc*t)
    out = out - out.mean()
    
    outTonic = numpy.exp(-t/tonicTau)
    
    cs1 = out/numpy.max(numpy.cumsum(out))
    cs2 = tonicA*outTonic/numpy.max(numpy.cumsum(outTonic))
    
    return cs1 + cs2


def filtertime(stim, Kt):
    '''  do the time filtering
   stim ... stimulus [xdim x ydim x tdim]
   kt   ... temporal filter kernel
   '''
    nTimeSteps = stim.shape[2]
    
    #out = numpy.zeros(numpy.r_[stim.shape[0:2], nTimeSteps+len(Kt)-1])
    #for x in range(stim.shape[0]):
    #    for y in range(stim.shape[1]):
    #        out[x,y,:] = scipy.signal.convolve(stim[x,y,:], Kt)

    # half the time 
    Ktre = numpy.zeros((1, 1, len(Kt)))
    Ktre[0,0,:]  = Kt[:]
    outre = scipy.signal.convolve(stim, Ktre) 

    # appearently only round off errors...
    #return out
    return outre


def filtertime_interp(stim, Kt, dt, Tstim):
    '''  do the time filtering
   stim ... stimulus [xdim x ydim x tdim]
   kt   ... temporal filter kernel
   '''
    
    if len(stim.shape)==2:
        stim = stim[:,:,numpy.newaxis]

    nstm = stim.shape[2]
    dstm = Tstim/float(nstm)    

    nt = int(numpy.floor(Tstim/dt))

    if nstm==1:
        out = numpy.tile(stim,(1,1,nt))*numpy.sum(Kt)
        return out

    if not numpy.sum(stim.flatten()):
        # only spont act no stim
        out = numpy.zeros(numpy.r_[stim.shape[0:2], nt])
        return out
    
    #to exclude artefacts
    nK = Kt.size
    nartstm = int(numpy.ceil(nK*dt/dstm)) 
    z = numpy.ones(nstm).astype(numpy.int);
    z[0] = nartstm + 1
    rstim = numpy.repeat(stim,z,axis=2)

    tart = nartstm*dstm
    ntart = int(numpy.ceil(nartstm*dstm/dt)) 

    #array to note the stim changes
    tarr = numpy.arange(0,Tstim+tart,dt)
    stmmsk = numpy.zeros(nt+ntart)

    for i in range(nstm+nartstm):
        stmmsk += numpy.where(tarr>=dstm*i,1,0)

    out = numpy.zeros(numpy.r_[rstim.shape[0:2], nt+ntart])
    
    for t in range(nt+ntart):
        # compose correct filter
        K = Kt[0:min(t+1,len(Kt))]
        msk = stmmsk[t::-1][:len(K)]
        Kinterp = numpy.zeros(nstm+nartstm) 
        for i in range(nstm + nartstm):
            Kinterp[i] = numpy.sum(K[msk==(i+1)])

        #now convolution
        out[:,:,t] = numpy.sum(rstim[:,:,Kinterp<>0] * Kinterp[Kinterp<>0],2)
        
    return out[:,:,ntart:]


def convolve2dsame(v, k):
    '''  calls scipy.signal.convolve_2d, but the result has the same size as v
    '''

    #avoid artifacts attach border
    n2 = numpy.max(k.shape)/2
    z = numpy.ones(v.shape[0]).astype(numpy.int)
    z[0] = n2+1
    z[-1] = n2+1
    
    vrep = numpy.repeat(v,z.astype(numpy.int),axis=0)

    z = numpy.ones(v.shape[1]).astype(numpy.int)
    z[0] = n2+1
    z[-1] = n2+1

    vrep = numpy.repeat(vrep,z.astype(numpy.int),axis=1)
    
    o = scipy.signal.convolve2d(vrep, k, 'same')

    o = o[n2+1:-n2+1,n2+1:-n2+1].copy()
    
    #diffx = abs(v.shape[0] - k.shape[0])
    #diffy = abs(v.shape[1] - k.shape[1])
    #
    #if diffx or diffy:
    #    o = o[diffx/2+(diffx%2):o.shape[0]-diffx/2, diffy/2+(diffy%2):o.shape[1]-diffy/2]        

    return o



#-----------------------------------------------------------------------------
class RetinaException(Exception):
    pass


class Retina(object):
    def __init__(self, dims=numpy.array([50, 50]), inner=0.11, outer=0.33,
                 dpl=0.145, scale=1.0, thres=0.0, omega= 0.85,
                 substractMean=True,contrastNormalization=False,inverse=True):
        '''
        dims  ... size of the retina
        inner ... std of the inner gaussian (degree)
        outer ... std of the outer gaussian (degree)
        dpl   ... degree per lattice spacing
        scale ... kernel scaling
        thres ... retina nonlinearity
        inverse ... whether to enforce same OFF and ON cell response to inverse stimuli
        
        Default parameters from:
        GAZERAS, Borg-Graham, Y Fregnac, Visual Neuroscience (1998)
        
        
        One other set of parameters (DOG model fitted to cat data):
        EINEVOLL, 2000, Visual Neuroscience
        
        g(r) = scale*(1/pi/inner^2 exp(-r^2/inner^2) - omega/pi/outer^2 exp(-r^2/outer^2))
        
        scale = 129 spikes/sec (BUT: this will be scaled after LGN anyway)
        inner = 0.59 or 0.62 deg
        outer = 1.27 or 1.26 deg
        
        omega = 0.88 or 0.85
        
        
        NOTE: Tadmor & Tolhurst, Vision research (2000) suggest that
        ganglion cells are CONTRAST sensitive. Therefore it should be
        devided by the local constrast. This can be implemented as
        the sum of center surround activity.

        CAUTION: this normalization has a very high OFF activity at
        borders I dont know if this is realistic!!
        
        (contrastNormalization=True)
        
        '''      
        self.inner = inner
        self.outer = outer
        self.dpl = dpl
        
        self.scale = scale # scale of kernel
        self.thres = thres
        self.dims = dims
        self.omega = omega

        self.cluster = None

        self.k = None
        self.substractMean = substractMean # for DOG Kernel

        self.contrastNormalization  = contrastNormalization

        self.inverse = inverse

        
    def generate(self):
        """ sets kernel according to parameters"""
        # one could make a spaceHorizon to speed it up...
        
        if self.contrastNormalization:
            sepkern = True
            submean = False
        else:
            sepkern = False
            submean = self.substractMean
        
        K = DoG(self.dims, self.inner/self.dpl, self.outer/self.dpl, self.omega,
                submean,separateKernels=sepkern)

        if not sepkern:
            K = [K]

        self.k = []
        for i in range(len(K)):
            self.k.append(K[i].copy()*self.scale)
        
        # maybe offset OFF and ON (0.63deg)?? see Conway & Livingstone, j. Neurosci (2006)
        

    def __processStimulus(self,stim,K):
        """ core routine for processing"""

        nTimeSteps = stim.shape[2]

        out = numpy.zeros((self.dims[1], self.dims[0], nTimeSteps))

        if numpy.sum(stim.flatten()):  # not only spont act no stim
            if self.cluster is None:
                for t in range(nTimeSteps):
                    s = stim[:,:,t]

                    if s.shape[0] <> self.dims[0] or s.shape[1] <> self.dims[1]:
                        im=scipy.misc.toimage(s, mode='F')
                        im=im.resize(self.dims)
                        s=scipy.misc.fromimage(im)
                        
                    out[:,:,t]=convolve2dsame(s, K)

            else:           # use the cluster
#                from cluster.cluster import Cluster, ClusterConfig

                import ipython1.kernel.api as kernel
                rc = self.cluster.getRemoteControllerClient()
                tc = self.cluster.getTaskControllerClient()
                nengines =len(rc.getIDs())
                tids=[]
                rc.resetAll()

                smallamount = self.dims[0]<40
                if smallamount:
                    #do directly
                    rc.executeAll('import scipy\nimport inputs.lgn as l\n')                    
                    rc.pushAll(k=K, dims=self.dims)
                
                    cmd = '''
if s.shape[0] <> dims[0] or s.shape[1] <> dims[1]:
    im=scipy.misc.toimage(s, mode='F')
    im=im.resize(dims)
    s=scipy.misc.fromimage(im)

o=l.convolve2dsame(s, k)
'''

                    for t in range(nTimeSteps):
                        setupNS = dict(s = stim[:,:,t])
                        tids.append(tc.run(kernel.Task(cmd, resultNames=['o'], setupNS=setupNS)))
                    
                    logger.info('%d tasks started' % len(tids))
                    tc.barrier(tids)
                    for tid in range(len(tids)):
                        res = tc.getTaskResult(tids[tid])
                        out[:,:,tid] = res.results['o']

                else:
                    #large amount of data (exceeding cluster submit size)
                    tmpfname = os.environ['HOME'] + '/__tmp_retina_processing'
                    rc.executeAll('import scipy\nimport os\nimport inputs.lgn as l\nimport scipy.io')                    
                    rc.pushAll(dims=self.dims,s=[],k=[],o=[],v=[])

                    cmd = '''
v= scipy.io.loadmat(fname)
s= v['s']
k= v['k']
os.remove(fname+'.mat')


if s.shape[0] <> dims[0] or s.shape[1] <> dims[1]:
    im=scipy.misc.toimage(s, mode='F')
    im=im.resize(dims)
    s=scipy.misc.fromimage(im)

o=l.convolve2dsame(s, k)
scipy.io.savemat(fname,{'o':o})
'''

                    for t in range(nTimeSteps):
                        fname = tmpfname + str(t) 
                        scipy.io.savemat(fname,{'s':stim[:,:,t],'k':K})
                        setupNS = dict(fname=fname)
                        tids.append(tc.run(kernel.Task(cmd, resultNames=[], setupNS=setupNS)))

                    logger.info('%d tasks started' % len(tids))

                    tc.barrier(tids)
                    os.system('ls -l >/dev/null')

                    for tid in range(len(tids)):
                        res = tc.getTaskResult(tids[tid])
                        
                        if None<>res.failure.printDetailedTraceback():
                            res.failure.raiseException()

                        fname = tmpfname + str(tid) 
                        out[:,:,tid] = scipy.io.loadmat(fname)['o']
                        os.remove(fname + '.mat')
                    
        return out

    
    def __calcContrast(self,Sc,Ssur):
        #C = numpy.abs(Sc-Ssur)/(numpy.abs(Ssur) + 1e-16)
        C = numpy.abs(Sc-Ssur)/(numpy.abs(Ssur) +numpy.abs(Sc)+ 1e-16)
        return C

        
    def processStimulus(self, stim):
        ''' do the spatial decorrelation
        stim ... stimulus [xdim, ydim, tdim]
        '''
        if self.k is None:
            self.generate()

        if stim.max() > 1.0 or stim.min() < 0.0:
            raise Exception('the intensity of the bitmaps must be between 0.0 and 1.0')

        pout = []
        for i_k in range(len(self.k)):

            pout.append(self.__processStimulus(stim,self.k[i_k]))

            if self.inverse:
                pout.append(self.__processStimulus(1-stim,self.k[i_k]))

        if not self.contrastNormalization:
            if not self.inverse:
                Son = numpy.abs(pout[0] * (pout[0] > self.thres))
                Soff = numpy.abs(-pout[0] * (pout[0] < self.thres))
            else:
                #off explicit
                Son = numpy.abs(pout[0] * (pout[0] > self.thres))
                Soff = numpy.abs(pout[1] * (pout[1] > self.thres))

            return (Son, Soff)

        else:
            #local contrast
            #contrast by difference in reposen of cewnter and surround
            # we have in general: R(rho) = C*(int S(r-rho)(G_c(r) -
            # G_s(r))) (where S is stim and C contrast). See
            # e.g. Croner & Kaplan 1995, Enroth-Cugell & Robson 1966,
            # Rodieck 1965

            if not self.inverse:
                Scenter = pout[0]
                Ssurround = pout[1]
                C = self.__calcContrast(Scenter,Ssurround/self.omega)

                Son = C*(Scenter - Ssurround)
                Son = Son * (Son>0)

                Soff = C*(Ssurround - Scenter) 
                Soff = Soff * (Soff>0)

            else:
                #ON center
                Scenter = pout[0]
                Ssurround = pout[2]

                C = self.__calcContrast(Scenter,Ssurround/self.omega)

                Son = C*(Scenter-Ssurround)
                Son = Son * (Son>0)

                #OFF center
                Scenter = pout[1]
                Ssurround = pout[3]

                C = self.__calcContrast(Scenter,Ssurround/self.omega)
                
                Soff = C*(Scenter-Ssurround)
                Soff = Soff * (Soff>0)

            return (Son,Soff)

             
    def plot(self):
        """ plot kernel"""
        import pylab
        from utils.mscfuncs import imagesc
        for i in range(len(self.k)):
            imagesc(self.k[i])
            pylab.colorbar()           

#-----------------------------------------------------------------------------

class LGNException(Exception):
    pass


class LGN(object):
    # default values:
    # 1.61 x 1.61 mm2 retina
    # 7.142 x 7.142 degree visual field
    # (so that 5 x 5 degree letter size)
    # 2500 retinal X-cells
    # 10000 LGN cells
    # 0.145 degree / retinal grid spacing
    # see Troyer et al. 1998 G. (Orbans book)

    def __init__(self, wc=None, thres=0, ret=Retina(),
                 fspot=160, fbackground=0.5, 
                 gamma_freq_switch = numpy.array([0, 30]), gamma_r = numpy.array([1, 5]),
                 dt=0.001, cluster=None, lgnKernel=LgnDongTonicKernel,lgnKernelArgs={},
                 dims = None, lagged=True):
        '''  thres             ... std of the inner gaussian (degree)
        wc                ... kernel cutoff !! DEPRECEATED use lgnKernel function instead
        ret               ... a Retina
        fspot             ... response to an optimal centered spot
        fbackground       ... spontaneous (poisson) activity of LGN cells [HZ]
        gamma_freq_switch ... frequency switches for gamma_r
        gamma_r           ... Regularity parameter at different gamma_freq_switch
        
        dims              ... dimension of LGN (centered at retina dims)
                              None takes dims from retina
        cluster           ... cluster object to use cluster for processing 

        dt    ... lgn time filter resolution (NEW as of 26.12.2007 MJR)
        '''

        self.thres = thres
        self.ret = ret

        if wc is not None:
            print "WARNING: 'wc' argument for Lgn is depreciated!\nuse 'lgnKernel=MyLGNKernelFun' (and lgnKernelArgs={} for optional args) instead"
        self.wc = wc
        self.lgnKernel = lgnKernel
        self.lgnKernelArgs = lgnKernelArgs
        
        self.lgnKernelHorizon = 0.5          # when the contribution of kernel is negligible

        self.fspot = fspot
        self.fbackground = fbackground

        self.gamma_freq_switch = gamma_freq_switch
        self.gamma_r = gamma_r

        #self.Tstim = Tstim  #!!!!!!!set in processStimulus
        self.dt = dt

        self.k = None
        self.analogLGNoutput = None
        self.shape = None

        self.dims = dims
        
        self.cluster = cluster
        
        self.lagged = lagged
        if self.lagged:
            # 4 types of LGN cells (on_nonlagged, on_lagged, off_nonlagged, off_lagged)
            self.nLgnLayers = 4 #fixed
        else:
            self.nLgnLayers = 2
            
        

    def processStimulus(self, stim, Tstim=None,dt=None):
        ''' processes the stimulus (calculates the firing frequencies)
        stim  ... stimulus [xdim, ydim, tdim]
        Tstim ... simulatioan time
        dt    ... lgn time filter resolution (NEW as of 26.12.2007 MJR)
        '''
        if Tstim:
            self.Tstim = Tstim
        else:
            raise LGNException(' provide Tstim!')

        if self.cluster!=None:
            self.ret.cluster = self.cluster
            
        #Note: stim is given in dim_t samples in time. These will be
        #      stretched to cover the whole Tstim. Retina does not depend on time,
        #      so stretching will be done ofter retina processing
        if dt:
            self.dt = dt;

        logger.debug('LGN.processStimulus(): start LGN process stim')
        tti = time.time()
        logger.info('start LGN.processStimulus')

        self.nTimeSteps = numpy.floor(float(self.Tstim)/self.dt)
        if self.dt > 10e-3:
            print ' WARNING: dt is rather big! (', self.dt,'sec) Think of decreasing it..'

        self.ret.generate()

        #RETINA PROCESSING: do spatial decorrelation
        tt = time.time()
        logger.debug('LGN.processStimulus(): start Retina.processStimulus()')

        (Son_short, Soff_short) = self.ret.processStimulus(stim)
        logger.debug('LGN.processStimulus(): finished Retina.processStimulus() (t=%1.4f sec)' % (time.time()-tt))

        if self.dims==None:
            self.dims = self.ret.dims

        elif not numpy.all(self.dims==self.ret.dims):
            #get dimensions right
            b = (numpy.array(self.ret.dims) - self.dims)

            if numpy.any(b<0):
                raise LGNException(" LGN size must be smaller or equal the visual field!")

            b = b / 2
            Son_short = Son_short[b[0]:b[0]+self.dims[0],b[1]:b[1]+self.dims[1],:]
            Soff_short = Soff_short[b[0]:b[0]+self.dims[0],b[1]:b[1]+self.dims[1],:]
            
        #LGN PROCESSING:  do temporal decorrelation
        t=numpy.arange(0, self.lgnKernelHorizon+self.dt/2.0, self.dt)

        if self.wc != None:
            self.k = LgnKernel(t,self.wc)
        else:
            self.k = self.lgnKernel(t,**self.lgnKernelArgs) #kernel fun defined

        logger.debug('LGN.processStimulus(): start LGN.filtertime')
        tt = time.time()
        
        #new version MJR 26.12.2007
        # NOTE: this will be MUCH faster if stimulus does not change fast (i.e. still images)
        # otherwise one could consider taking the old version again (i.e. for movies)
        # but also in the latter case this version seems to be OK

        # THEREFORE: stmframes should really only have an index for changes in stimuli.
        # i.e. [1,0] and Tstim=1 will show the first stim for 500ms and the second stim for the rest
        # (see also v1test for an example)

        S_lgn_on = filtertime_interp(Son_short, self.k,self.dt,self.Tstim)
        S_lgn_off = filtertime_interp(Soff_short, self.k,self.dt,self.Tstim)

        logger.debug('LGN.processStimulus(): finished LGN.filtertime (t=%1.4f sec)' % (time.time()-tt))
        logger.debug('LGN.processStimulus(): post-process')

        if self.lagged:
            S_on_nlag = S_lgn_on * (S_lgn_on > self.thres) 
            S_off_nlag = S_lgn_off * (S_lgn_off > self.thres)

            S_on_lag = -S_lgn_on * (S_lgn_on < self.thres)
            S_off_lag = -S_lgn_off * (S_lgn_off < self.thres)

#            S_lst = [S_on_nlag, S_on_lag, S_off_nlag, S_off_lag]
            S_lst = [S_on_nlag, S_off_nlag, S_on_lag, S_off_lag]
        else:
            # think about this!!!
            S_on_nlag = S_lgn_on * (S_lgn_on > self.thres)
            S_off_nlag = S_lgn_off * (S_lgn_off > self.thres)

            S_lst = [S_on_nlag, S_off_nlag]

        analogLGNoutput = numpy.zeros(numpy.r_[self.nLgnLayers, S_on_nlag.shape])

        for i in range(len(S_lst)):
            j = min(i, self.nLgnLayers-1)
            analogLGNoutput[j,:,:,:] = S_lst[i]

        nTimeSteps = analogLGNoutput.shape[3]

        # info
        self.shape = numpy.r_[analogLGNoutput.shape[:3],self.nTimeSteps]

        analogLGNoutput = numpy.reshape(analogLGNoutput, numpy.r_[self.nLgnLayers*numpy.prod(analogLGNoutput.shape[1:3]), nTimeSteps])
        analogLGNoutput = analogLGNoutput[:,:self.nTimeSteps]

        # set input rate normalization factor so that optimal
        # stimulation frequency of an ON nonlagged cell is fspot

        if self.ret.k is None:        # should never occur!!!
            raise RetinaException('Retina kernel not yet generated!\nrun retinaProcess stimulus first')

        self.analogLGNoutput_unscaled = analogLGNoutput.copy()
        if not self.ret.contrastNormalization:
            K = self.ret.k[0]
            max_space_response = K[K > 0.0].sum()
        else:
            CK = self.ret.k[0] - self.ret.k[1]
            mxkern = CK[CK>0.0].sum()

            h1 = self.ret.k[0][CK>0.0].sum()
            h2 = self.ret.k[1][CK>0.0].sum()/self.ret.omega
            #cmax = (h1/h2-1)
            cmax = numpy.abs(h1-h2)/(h2+h1)

            max_space_response = cmax*mxkern

        max_time_response = max(numpy.cumsum(self.k))
        fspotmax = max_space_response * max_time_response;

        # scale input rates to fspot for optimal spot stimulation
        analogLGNoutput /= fspotmax
        analogLGNoutput *= (self.fspot - self.fbackground) 
        analogLGNoutput += self.fbackground

        self.analogLGNoutput = analogLGNoutput.copy()

        logger.info('maximal output %1.3f Hz (fspot=%1.3f Hz)' % (analogLGNoutput.max(),self.fspot))

        logger.info('LGN.processStimulus() finished (t=%1.2f sec)' % (time.time()-tti))


    def generate(self):
        ''' generates the output spike trains of the LGN '''

        logger.info('start LGN generate')

        tt = time.time()
        if self.k is None:
            raise LGNException('call method processStimulus() first!')

        cl=self.cluster

        self.gamma = gammagen.GammaProcessRate(f=self.analogLGNoutput, Tstim=self.Tstim,
                                      fbackground=self.fbackground, 
                                      gamma_freq_switch=self.gamma_freq_switch, 
                                      gamma_r=self.gamma_r, cluster=cl)
        g = self.gamma.generate()
        logger.debug('LGN.generate(): LGN generate finished (t=%1.4f sec)' % (time.time()-tt))
        logger.info('LGN.generate finished (t=%1.2f sec)' % (time.time()-tt))
        return g


    def plot(self, gap=0):
        """ Plots analog output as single imagescale"""

        if self.k is None:
            return

        import pylab
        from utils.mscfuncs import imagesc
        
        pylab.figure()

        X = self.analogLGNoutput.reshape(self.shape,order='C')

        for i in range(X.shape[0]):
            x =X[i,:,:,::(gap+1)]

            x = x.transpose([2,1,0])
            sz = x.shape
            x = numpy.reshape(x,numpy.r_[sz[0]*sz[1],sz[2]]).transpose()

            pylab.subplot(X.shape[0],1,i+1)

            imagesc(x)
            pylab.title('LGN layer %d' % (i+1));


    def plotSeq(self, n=100, layer=0, cmap=None):
        """ Plots analog output as sequence of pictures."""
        import pylab
        from utils.mscfuncs import imagesc

        if self.k is None:
            return

        X = self.analogLGNoutput.reshape(self.shape,order='C')

        if not type(layer)==list:
            layer = [layer]

        r = numpy.int(numpy.ceil(n/numpy.sqrt(n)))
        indarr = numpy.linspace(0,X.shape[3]-1,n)

        mx = numpy.max(X[:])
        if layer[0]!=None:
            for i in layer:
                pylab.figure()
                dt = self.dt

                s = 0
                for j in indarr:
                    jj = numpy.floor(j)
                    s = s+1
                    pylab.subplot(r,r,s)
                    imagesc(X[i,:,:,jj],scale=[0,self.fspot/2.],cmap=cmap)
                    pylab.title('%1.2f sec' % (jj*self.dt),fontsize=6 )
                    pylab.axis('off')
            pylab.colorbar()
        else:
            #plot all layers
            pylab.figure()
            layername = ['ON','ON lagged','OFF','OFF lagged']
            layer = [0,1,2,3]
            s = 0
            for i in layer:
                for j in indarr:
                    s = s+1
                    jj = numpy.floor(j)

                    pylab.subplot(len(layer),n,s)
                    imagesc(X[i,:,:,jj],scale=[0,self.fspot/2.],cmap=cmap)
                    pylab.title('%1.2f sec' % (jj*self.dt),fontsize=6 )
                    pylab.axis('off')

                    if j==indarr[0]:
                        pylab.subplot(len(layer),n,i*n+1)
                        pylab.ylabel(layername[i])

                
    def plotTimeLine(self, idx=(0,0)):
        """ Plots Time of analog output."""
        import pylab
        if self.k is None:
            return
        
        pylab.figure()
        X = self.analogLGNoutput.reshape(self.shape,order='C')
        x = X[:,idx[0],idx[1],:]

        t = numpy.arange(0,self.Tstim,self.dt)
                
        pylab.plot(t,x.transpose())
        #pylab.legend(('On non-lagged cell','ON lagged cell','OFF non-lagged cell', 'OFF lagged cell'))
        if self.lagged:
            pylab.legend(('On non-lagged cell','OFF non-lagged cell','ON lagged cell', 'OFF lagged cell'))
        else:
            pylab.legend(('On cell','OFF cell'))

        pylab.xlabel('Time [s]')
        pylab.ylabel('Analog Rate Output')
        pylab.title('Time line of one pixel x=%d y=%d' % idx)


    def plotFreqResponse(self, stim, ind=None, s=None, num_steps=10, colorbarif=True, retinaif=False):
        '''plot the output'''
        import pylab
        pylab.figure()

        if ind is None:
            ind = range(0, stim.shape[2], numpy.ceil(float(stim.shape[2])/num_steps))

        V = concBitmap(stim, ind)
        if self.lagged:
            tit = ['nonlagged on cells', 'nonlagged off cells', 'lagged on cells', 'lagged off cells']
        else:
            tit = ['nonlagged on cells', 'nonlagged off cells']

        pylab.subplot(len(tit)+1,1,1)
        pylab.imshow(V, cmap=pylab.cm.gray)
        if colorbarif: pylab.colorbar()
        pylab.title('stimulus')
        pylab.xticks([]); pylab.yticks([])

        data=self.analogLGNoutput.reshape(self.shape, order='C')
        for k in range(len(tit)):
            V = concBitmap(data[k,:,:,:], ind)
            pylab.subplot(len(tit)+1,1, 2+k)
            pylab.imshow(V)
            if colorbarif: pylab.colorbar()
            pylab.title(tit[k])
            pylab.xticks([]); pylab.yticks([])


    def setStimulusPos(self, response):
        ''' 
        '''        
        response.pos_x=[]
        response.pos_y=[]
        response.pos_z=[]

# does not work, because channel.idx = -1
#        for c in response.channel:
#            response.pos_z.append(int(c.idx) / int(self.dims.prod()))
#            idx = c.idx % self.dims.prod()
#            response.pos_x.append(idx / self.dims[1])
#            response.pos_y.append(idx % self.dims[1])

        for i in range(len(response.channel)):
            response.pos_z.append(int(i) / int(self.dims.prod()))
            idx = i % self.dims.prod()
            response.pos_x.append(idx / self.dims[1])
            response.pos_y.append(idx % self.dims[1])
        
        response.pos_x=numpy.array(response.pos_x)
        response.pos_y=numpy.array(response.pos_y)
        response.pos_z=numpy.array(response.pos_z)
        response.save_attrs += ['pos_x','pos_y','pos_z']
#----------------------------------------------------------------------------

class LGNEphysGammaBackground(LGN):
    """ overloads LGN. Loads LGN rate distribution from monkey data
    and assigns fixed rates randomly for all cells (ON/OFF/LAGGED). No
    filtering (therfore no correlation between ON/OFF), but
    gammaprocessrate will be applied"""

    def __init__(self,  stmtype='spont', area='LGN', rorc=0.,  **kwds):
        super(LGNEphysGammaBackground, self).__init__(lagged=False,**kwds)

        self.rorc = rorc #constant if 0

        self.loadpath = default.dirs['ephys']

        self.k = 'dummy'                # for plotting
        self.stmtype = stmtype
        self.area = area

    
    def processStimulus(self, stim, Tstim=None, dt=None):
        """overloads processstim of LGN. STIM and DT have no function at all (just to make it compatible) """

        logger.info('start LGNRealSpontanousBackground')
        tt = time.time()

        if Tstim:
            self.Tstim = Tstim
        else:
            raise LGNException(' provide Tstim!')

        if dt:
            self.dt = dt;

        if self.dims==None:
            self.dims = self.ret.dims

        #get rates
        L = int(self.nLgnLayers * numpy.prod(self.dims)) # only for non lagged
        errate = ephysrate.EphysRandomRate(Tstim=self.Tstim, nChannels=L,dt=self.dt,area=self.area,stmtype=self.stmtype,rorc=self.rorc)

        #calc the rates
        Rmat = errate.calcRate()
        
        self.nTimeSteps = Rmat.shape[1];
        self.shape = numpy.r_[self.nLgnLayers, self.dims , self.nTimeSteps].tolist()

        R = Rmat.reshape(numpy.r_[self.nLgnLayers, self.dims, self.nTimeSteps], order='f')

        self.analogLGNoutput = numpy.reshape(R, numpy.r_[self.nLgnLayers*numpy.prod(R.shape[1:3]), self.nTimeSteps])

        logger.info('LGNRealSpontanousBackground processStimulus finished (t=%1.2f sec)' % (time.time()-tt))

#-----------------------------------------------------------------------------


if __name__ == '__main__':
    import pylab
    import inputs.bitmapstim as bstm
    import numpy
    from utils.mscfuncs import imagesc

    dim_lgn=numpy.array([50, 50])

    #bmp = bstm.makeBitmap(dim_lgn, 'A')
    #pylab.imshow(bmp)

    Tstim=0.6

    #lgn = LGNEphysGammaBackground(dims=dim_lgn,rorc=0.5)

    #lgn.processStimulus([],Tstim,0.001)
    #input = lgn.generate()

    #lgn.plotSeq()

    #pylab.figure()
    #input.plot()

    #pylab.show()


    if 1:           
        stmframes = numpy.r_[0, 1, 1, 1, 0, 0]
        letters = ['B','A','E']


        dims = numpy.r_[dim_lgn, len(stmframes)]
        stim = bstm.generateLetterStim(dims, stmframes, letters)
        s = stim.shape

        #stim = stim + numpy.random.rand(s[0],s[1],s[2])>0.8
        #stim[stim>1] = 1
        #stim = 1 - stim

        if 0:
            t=numpy.arange(0, 0.3, 1e-3)
            KtDongTonic=LgnDongTonicKernel(t)

            inv = False
            sin = 0.05
            sout = 0.3
            om = 0.65
            lgn = LGN(ret=Retina(dims=dim_lgn,inverse=inv,contrastNormalization=False,inner=sin, outer=sout,omega=om),dt=0.01)
            lgn.processStimulus(stim,Tstim)
            
            lgn_contrast = LGN(ret=Retina(dims=dim_lgn,inverse=inv,contrastNormalization=True,inner=sin, outer=sout,omega=om),dt=0.005)
            lgn_contrast.processStimulus(stim,Tstim)


        else:
            import paramsearch.experiments as e
            pars = e.getStandardPars()
            lgn_contrast = pars.getLGN()
            lgn_contrast.processStimulus(stim,Tstim)
            
        #lgn.plotSeq(10)
        #lgn.plotSeq(10,layer=2)


        lgn_contrast.plotSeq(13,layer=None)
        #lgn_contrast.plotSeq(49,layer=1)
        #lgn_contrast.plotSeq(49,layer=2)
        #lgn_contrast.plotSeq(49,layer=3)


        pylab.show()
