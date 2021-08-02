from inputs.spikes2rate import *
from datetime import datetime

import numpy
import scipy
import scipy.signal
import dataformats
import matplotlib
import matplotlib.mlab
import scipy.weave as wv
from pyV1.utils.mscfuncs import i2s,s2i



def restrictSpikeLst(spikelst,tregion):
    """deletes spike not in the interval"""

    def recursiveListInterval(lst,interval):
        """ recursively excludes incorrect spikes"""
        if (type(lst)==list) or (type(lst)==numpy.ndarray and  (lst.dtype==numpy.dtype('object'))):
            for i in range(len(lst)):
                lst[i] = recursiveListInterval(lst[i],interval)
        else:
            if type(lst)==type(numpy.array([])):
                arr = lst[numpy.where(lst>=interval[0])[0]]
                arr = arr[numpy.where(arr<interval[1])[0]]
                lst = arr
        return lst

    return recursiveListInterval(spikelst,tregion)


def getSpikeList(resp, startT=-numpy.inf, endT=numpy.inf, channels=None):
    '''given a Response object, return the list of spikes in the given time period'''
    if channels is None:
        return [chan.data[(chan.data>=startT) & (chan.data<=endT)] for chan in resp.channel]

    return [resp.channel[i].data[(resp.channel[i].data>=startT) & (resp.channel[i].data<=endT)] for i in channels]


def getCombindedSpikeList(resp, startT=-numpy.inf, endT=numpy.inf, channels=None):
    '''given a Response object, return the array of all spikes in the given time period'''

    spikes=numpy.concatenate(getSpikeList(resp, startT, endT, channels))
    spikes.sort()

    return spikes


def getEleSpikeList(resp,munits,startT=-numpy.inf, endT=numpy.inf,maxele=1000,channels=None):
    """ retuns a pseud Electrode signal, where channels are combined
    (number of chanels combined is based on a poisson distribution
    with mean munits).
    channels are chosen WITHOUT replacement!
    NO distance dependence! """

    if channels==None:
        idx = range(len(resp.channel))
    else:
        idx = channels

    idx = numpy.array(idx)
    ele_lst = []
    p = numpy.random.permutation(len(idx)).astype(numpy.int)

    s_start = 0
    s_end = 0
    for n in range(maxele):
        nunits = numpy.random.poisson(munits)+1;
        s_end += nunits

        if s_end>len(idx):
            break

        curidx = idx[p[s_start:s_end]]
        s_start = s_end
        spks = numpy.concatenate([resp.channel[i].data[(resp.channel[i].data>startT) & (resp.channel[i].data<endT)] for i in curidx])
        ele_lst.append(numpy.unique(spks)) # sort and unique

    return ele_lst


def getEleSpikeListDist(resp,munits,startT=-numpy.inf,endT=numpy.inf,nele=1000,
                        layer_base=None,layer_depth=[3,3,3],forcepos=None,zsigmafrac=1.,seed=None):
    """ retuns a pseudo electrode signal, where channels are combined
    (number of channels combined (nunits) is based on a Poisson distribution
    with mean munits).

    NELE electrodes are choosen with random position. NUNITS nearby
    channels will be combined (NUNITS from Poisson).

    Note that in principle channels are chosen WITH replacement!

    FORCEPOS forces the electrode position to be in the given z-layers
    (note that usually layer 2/3 has highest indices (6,7,8), layer
    5/6 lowest (0,1,2))

    """

    numpy.random.seed(seed)

    #get grid
    if layer_base==None:
        #assume square
        nx = numpy.sqrt(float(numpy.sum(resp.layeridx==0))/float(layer_depth[0]))
        if int(numpy.floor(nx)) != int(nx):
            raise Exception('not square. provide layer_base')

        layer_base = [nx,nx]

        

    nzl = numpy.array(layer_depth)
    nx = int(layer_base[0])
    ny = int(layer_base[1])
    
    nl = numpy.prod(layer_base)*nzl
    cnl = nl.cumsum()-nl[0]
    
    n = len(resp.channel)

    nele = numpy.min([nele,n])

    s = nzl.size-1
    layidx = []
    for l in nzl[::-1]:
        layidx.append(s * numpy.ones(l))
        s -=1
    layidx = numpy.concatenate(layidx)
    
    if type(forcepos)==str:
        idx = numpy.where(numpy.array(resp.layers)==forcepos)[0][0]
        forcepos = numpy.arange(layer_depth[idx]).astype(numpy.int)
        forcepos = (forcepos + nzl[::-1][:len(nzl)-idx-1].sum()).tolist()


    eleidx = []
    eleidx = numpy.random.permutation(n)
        
    eleidx = eleidx[:nele] 
    

    mv = numpy.random.multivariate_normal;
    I = numpy.eye(3);

    I[0,0] = zsigmafrac;
    
    m = numpy.zeros(3)
    
    ele_lst = []
    elepos_lst = []
    for k in range(nele):
        nunits = numpy.min([numpy.random.poisson(munits)+1,n])

        if zsigmafrac>0:
            sig = (4*nunits)**(2./3.)/(2.*numpy.pi)/(zsigmafrac**(1./3.))
        else:
            sig = (4*nunits)/(2.*numpy.pi)

        lay = resp.layeridx[eleidx[k]]
        withinlayidx = eleidx[k]-nl[:lay].sum()
        posele = i2s([nzl[lay],ny,nx],withinlayidx)
        posele[0,0] +=  nzl[::-1][:len(nzl)-lay-1].sum()


        if forcepos <> None and not posele[0,0] in forcepos:
            posele[0,0] = forcepos[numpy.random.random_integers(0,len(forcepos)-1)]

        elepos_lst.append(numpy.atleast_2d(posele[0,::-1]))
        
        chidx_lst = []
        while len(chidx_lst)<nunits:

            #get indices from multvar gauss (a bit simple but effective!?)
            chsub = (numpy.floor(mv(m,sig*I,2*nunits)) + posele).astype(numpy.int)
            
            #nothing toroid
            chsub[chsub[:,0]<0,0] =0
            chsub[chsub[:,0]>=nzl.sum(),0] = nzl.sum()-1
            chsub[chsub[:,1]<0,1] =0
            chsub[chsub[:,1]>=ny,1] = ny-1
            chsub[chsub[:,2]<0,2] =0
            chsub[chsub[:,2]>=nx,2] = nx-1

            #get true indices
            chlay =  (len(nzl)-1)*numpy.ones(chsub.shape[0]).astype(numpy.int)
            for l in nzl[::-1].cumsum():
                w = numpy.where(chsub[:,0]-l>=0)[0]
                chlay[w] = chlay[w] - 1


            for s in range(len(nzl)):
                
                w = numpy.where(chlay==s)[0]
                if not len(w):
                    continue
                
                chsub[w,0] = chsub[w,0] - nzl[::-1][:len(nzl)-s-1].sum()
                #chsub[w,0] = nzl[s]-1 - chsub[w,0]
                
                chidx = s2i([nzl[s],ny,nx],chsub[w])
                chidx  += nl[::-1][:s].sum()
                chidx_lst = (numpy.r_[chidx_lst,chidx]).astype(numpy.int)



        chidx_lst = chidx_lst[:nunits]

        spks = numpy.concatenate([resp.channel[i].data[(resp.channel[i].data>startT) & (resp.channel[i].data<endT)] for i in chidx_lst])
        ele_lst.append(numpy.unique(spks)) # sort and unique

        dinfo = {}
        dinfo['elepos'] = numpy.concatenate(elepos_lst,axis=0)
        dinfo['layidx'] = layidx
        dinfo['layer_depth'] = layer_depth
        dinfo['layer_base'] = layer_base
        dinfo['layers'] = resp.layers
        dinfo['forcepos'] = forcepos
        dinfo['zsigmafrac'] = zsigmafrac
        dinfo['munits'] = munits

    return (ele_lst,dinfo)



def getSDF(spikes, dt, startT, endT, sigma=None):
    '''
    '''
    N = numpy.floor((endT-startT)/dt)
    ind = numpy.floor((spikes-startT)/dt).astype(numpy.int)
    
    sdf = subSDF(ind, N, dt, sigma)
        
    return sdf


def subSDF(ind, N, dt, sigma):
    '''
    '''
    ind = ind[ind<N]
    sdf = numpy.zeros(N)
    for i in ind:
        sdf[i]+=1

    if sigma is not None:
        isigma = numpy.ceil(sigma/dt)
        lenG = 6*isigma
        G = scipy.signal.gaussian(lenG, isigma)
        sdf = scipy.signal.convolve(sdf, G)
        sdf = sdf[lenG/2:-(lenG/2-1)]
        
    return sdf
   
def firingRate(resp, bin=0.2, startT=0.0, endT=1.0, dt=None, channels=None):
    ''' compute the firing rates
    '''

    spikes = getSpikeList(resp, startT, endT, channels=channels)
    nChannel = len(spikes)
    rates = []

    if dt is None:
        rates = [getSDF(spikes[i], bin, startT, endT)/bin for i in range(nChannel)]
    else:
        for i in range(int(bin/dt)):
            spks = [s-i*dt for s in spikes]
            rates += [getSDF(spks[i], bin, startT, endT)/bin for i in range(nChannel)]

    return numpy.concatenate(rates)


def firingRateCount(resp, startT=0.0, endT=None, channels=None):
    ''' compute the firing rates
    '''
    if endT is None:
        endT=resp.Tsim
    dt=endT-startT
    spikes = getSpikeList(resp, startT, endT, channels=channels)
    nChannel = len(spikes)

    return numpy.array([len(spikes[i])/dt for i in range(nChannel)])


def meanRate(resp):

    nChannel = float(len(resp))

    rates = [len(ch.data)/resp.Tsim for ch in resp.channel]
    return (numpy.mean(rates), numpy.std(rates)/numpy.sqrt(nChannel))

#================================================================================
# Note: calc.. function have to work on trials lists!


def calcRate(spiketimes):
    """ rates """

    if type(spiketimes)==type(numpy.array([1.])):
        tmp = spiketimes.ravel().tolist()
        tmp = [x for x in tmp if not len(x) or (len(x) and not numpy.isnan(x[0]))]
    else:
        tmp = spiketimes


    def firstlastrate(spks):

        if len(spks)>1:
            return float(len(spks)-2)/(spks[-1]-spks[0])
        else:
            return 0.

    
    tmp = numpy.array(map(firstlastrate,tmp))
    return   tmp





def calcISI(spiketimes):
    ''' the inter-spike-intervalls
    '''
    if type(spiketimes)==type(numpy.array([1.])):
        tmp = spiketimes.ravel().tolist()
        tmp = [x for x in tmp if not len(x) or (len(x) and not numpy.isnan(x[0]))]
    else:
        tmp = spiketimes
    
    return numpy.concatenate(map(numpy.diff,tmp))


def calc2ISI(spiketimes):
    ''' the inter-spike-intervalls
    '''

    if type(spiketimes)==type(numpy.array([1.])):
        tmp = spiketimes.ravel().tolist()
        tmp = [x for x in tmp if not len(x) or (len(x) and not numpy.isnan(x[0]))]
    else:
        tmp = spiketimes

    if not len(tmp):
        return numpy.array([[],[]]).T

    def subdiff2(spks):
        if len(spks)<2:
            return numpy.array([[],[]]).T
        else:
            df = numpy.diff(spks)
            return numpy.c_[df[0:-1],df[1:]]
        
    return numpy.concatenate(map(subdiff2,tmp),axis=0)



def calc2ISI2ELE(spiketimes,npairs=100,ntimes=1000,rate = 25):
    """ calculates the the 4 dimensional 2 isi for pairs of 2 eles, for at max npairs"""

    if type(spiketimes)==list:
        # assume that only elelst is given (in case of model)
        n = len(spiketimes)
        triallst = [spiketimes] # just one trial
        if type(spiketimes[0]) != type(numpy.array([])):
            raise Exception("Wrong type")

    elif type(spiketimes)==type(numpy.array([1])):
        # assume electrodes to be in the first dim
        sh = spiketimes.shape
        tmp = spiketimes.reshape(sh[0],numpy.prod(sh[1:])).transpose().tolist()


        triallst = []
        for trial in tmp:
            triallst.append([ele for ele in trial if not len(ele) or (len(ele) and not numpy.isnan(ele[0]))])

    else:
        raise Exception("Wrong type")

    res_lst = []
    for spklst in triallst:

        n = len(spklst)

        if not n:
            continue
        
        if npairs == None:
            m = n*(n-1)/2
        else:
            m = numpy.min([npairs,n*(n-1)/2])

        r_lst =[]
        M = 0
        while (M < m):
            tmp = numpy.random.random_integers(0,n-1,[m*2,2])
            # attention: 2 electrodes might be processed twice and

            #exclude non-cross term
            tmp = tmp[tmp[:,0]!=tmp[:,1]]

            M = M + len(tmp)
            r_lst.append(tmp)

        r2 = numpy.concatenate(r_lst)
        r2 = r2[:m]

        first = False
        for [i,j] in r2:
            spks1 = spklst[i];
            spks2 = spklst[j];

            if len(spks1)<2 or len(spks2)<2:
                continue
            mn = numpy.min([spks1.min(),spks2.min()])
            mx = numpy.min([spks1.max(),spks2.max()])

            isi1 = numpy.diff(spks1)
            isi2 = numpy.diff(spks2)

            spks1 = spks1[1:]
            spks2 = spks2[1:]

            T = mx-mn
            tarr = numpy.sort(numpy.random.rand(numpy.int(numpy.min([rate*T,ntimes])))*T+mn)

            nt = tarr.size;
            tmpres = numpy.zeros([nt,4]);
            n1 = len(spks1);
            n2 = len(spks2);
            s = 0;
            var_lst = ['tarr','nt','n1','n2','isi1','isi2','spks1','spks2','tmpres']

            code = '''

            int i,k,l,s;
            double t;
            k = 0;
            l = 0;
            s = 0;
            for (i=0;i<nt;i++) {
                t = tarr[i];
                while ((k<n1) && (t>spks1[k])) {k +=1;}
                while ((l<n2) && (t>spks2[l])) {l +=1;}

                /*printf("k=%d, spks1[k]=%1.2f, t=%1.2f \\n",k,spks1[k],t);
                printf("l=%d, spks2[l]=%1.2f, t=%1.2f \\n",l,spks2[l],t);*/
                if ((n1-1 > k+1) && (n2-1 > l+1)) {
                    tmpres[s] = isi1[k];
                    tmpres[s+1] = isi1[k+1];
                    tmpres[s+2] = isi2[l];
                    tmpres[s+3] = isi2[l+1];
                    s = s+4;
                }
                else {
                    break;
                }
            }

            return_val = s;
            '''
            s = wv.inline(code,var_lst)
            if s:
                if first:
                    res_lst.append([])
                res_lst.append(tmpres[:s/4,:].copy())
                first = False


    if len(res_lst)>1:
        res = numpy.concatenate(res_lst)
    elif len(res_lst)==1:
        res = res_lst[0]
    else:
        res = numpy.array([]);

    return  res



#=========================================================================


def getBinnedISI(spikes, bin, startT, endT):
    ''' compute the inter-spike-intervalls
    '''
    
    spikes = spikes[(spikes>startT) & (spikes<=endT)]
    T = endT-startT
    nBins = int(T/bin)
    isi=[]
    
    for i in range(nBins):
        s = spikes[(spikes>=i*bin) & (spikes<=(i+1)*bin)]
        isi.append(numpy.diff(s)[:-1])
    
    return numpy.concatenate(isi)


def isiDist(resp, bin=0.2, startT=0.0, endT=1.0, dt=None, channels=None):
    ''' compute the ISI
    '''
    
    spikes = getSpikeList(resp, startT, endT, channels=channels)
    nChannel = len(spikes)
    isi = []
        
    if dt is None:
        isi = [getBinnedISI(spikes[i], bin, startT, endT)/bin for i in range(nChannel)]
    else:
        for i in range(int(bin/dt)):
            spks = [s-i*dt for s in spikes]
            isi += [getBinnedISI(spks[i], bin, startT, endT)/bin for i in range(nChannel)]
    
    return isi


def isiCV(resp, bin, startT=-numpy.inf, endT=numpy.inf, dt=None):
    ''' compute the ISI CV 
    '''
    
    spikes = getSpikeList(resp, startT, endT)
    isidist=isiDist(resp, bin, startT, endT, dt)
    isicv = []
    
    for isi in isidist:
        isicv.append(isi.std()/isi.mean())
        
    return numpy.array(isicv)


def fanoFactorTime(resp, bin, startT=0.0, endT=1.0, channels=None):
    '''
    '''
    spikes = getSpikeList(resp, startT, endT, channels=channels)
    
    nChannel = len(spikes)
    
    sdf = [getSDF(spikes[i], bin, startT, endT) for i in range(nChannel)]
    fano = numpy.array([s.var()/s.mean() for s in sdf])
    
    return fano


def fanoFactor(resp, windows=numpy.logspace(-2,1,20), startT=0.0, endT=1.0, channels=None):
    '''
    '''
    return [fanoFactorTime(resp, bin, startT, endT, channels=channels) for bin in windows]


def meanFanoFactor(resp, windows=numpy.logspace(-2,1,20), startT=0.0, endT=1.0, channels=None):
    '''
    '''
    fano = [fanoFactorTime(resp, bin, startT, endT, channels=channels) for bin in windows]
    fano = [f[numpy.isnan(f)==False] for f in fano]

    return dict(data=numpy.array([f.mean() for f in fano]), error=numpy.array([f.std() for f in fano]), windows=windows)



def burstRate(resp, minBurst=2, maxBurst=20, max_isi=0.005, startT=0.0, endT=numpy.inf, channels=None):
    '''
    '''
    spikes = getSpikeList(resp, startT=startT, endT=endT, channels=channels)

    nChannel = len(spikes)
    N = numpy.arange(minBurst, maxBurst+1)
    bursts=numpy.array([(sum([calcBursts(spikes[i], n, max_isi) for i in range(nChannel)])) for n in N]).astype('float')

    bursts=bursts/nChannel

    T = endT-startT
    if not (numpy.isnan(T) or numpy.isneginf(T) or numpy.isinf(T)):
        bursts = bursts/T

    return dict(data=bursts, N=N)


def calcBursts(spikes, nSpikes, max_isi=0.005):
    '''
    '''
    min_burst = (nSpikes-1)*max_isi

    bursts = spikes[nSpikes-1:] - spikes[0:-(nSpikes-1)]
    bursts = bursts[bursts <= min_burst]

    return len((bursts[1:]-bursts[:-1]) > 0)


def spktdist_nonfunc(sp1, sp2, maxlag):
    lsp1 = len(sp1)
    lsp2 = len(sp2)
#    d = numpy.zeros(lsp1*lsp2)
    d = numpy.zeros(5)
    N=0
    code = """
        int sz1=d.size();
        int sz=d.size();
        for(int i=0; i<lsp1; i++) {
            for(int j=0; j<lsp2; j++) {
                double dist=sp2(j)-sp1(i);

                if(fabs(dist) <= maxlag) {
                    if (N >= sz)
                    {
                        d.resize(sz+sz1);
                        sz=d.size();
                    }
                    d(N)=dist;
                    N++;
                }
            }
        }
//        return_val = N;
        return_val = sz;
    """

    N=wv.inline(code, ['d', 'sp1','sp2','lsp1','lsp2','N','maxlag'],type_converters=wv.converters.blitz, compiler='gcc', verbose=1)

#    return d[:N]
    return d,N


def spktdist(sp1, sp2, maxlag):
    lsp1 = len(sp1)
    lsp2 = len(sp2)
    d = numpy.zeros(lsp1*lsp2)
#    d = numpy.zeros(5)
    N=0
    code = """
        double dist=0.0;
        for(int i=0; i<lsp1; i++) {
            for(int j=0; j<lsp2; j++) {
                dist=sp2(j)-sp1(i);

                if(fabs(dist) <= maxlag) {
                    d(N)=dist;
                    N++;
                }
            }
        }
        return_val = N;
    """

    N=wv.inline(code, ['d', 'sp1','sp2','lsp1','lsp2','N','maxlag'],type_converters=wv.converters.blitz, compiler='gcc', verbose=1)

    return d[:N]
#    return d,N


def getAllPairsUndirected(N):
    d = numpy.zeros((N**2,2), 'int')
    code = """
        int s=0;
        for(int i=0; i<N; i++) {
            for(int j=i; j<N; j++) {
                d(s,0)=i;
                d(s,1)=j;
                s++;
            }
        }
        return_val = s;
    """

    s=wv.inline(code, ['d','N',],type_converters=wv.converters.blitz, compiler='gcc', verbose=1)

    return d[:s]


def getAllPairs(N):
    d = numpy.zeros((N**2,2), 'int')
    code = """
        int s=0;
        for(int i=0; i<N; i++) {
            for(int j=i; j<N; j++) {
                d(s,0)=i;
                d(s,1)=j;
                s++;
            }
        }
        return_val = s;
    """

    wv.inline(code, ['d','N',],type_converters=wv.converters.blitz, compiler='gcc', verbose=1)

    return d


def getAllPairsNoAuto(N):
    d = numpy.zeros((N*(N-1),2), 'int')
    code = """
        int s=0;
        for(int i=0; i<N; i++) {
            for(int j=0; j<N; j++) {
                if(i!=j) {
                    d(s,0)=i;
                    d(s,1)=j;
                    s++;
                }
            }
        }
        return_val = s;
    """

    s=wv.inline(code, ['d','N',],type_converters=wv.converters.blitz, compiler='gcc', verbose=1)

    return d[:s]


def getPairs(nChannel, maxnpairs, auto=False, revert=False):
    if maxnpairs >= (nChannel**2)/2:
        if auto:
            pairs=getAllPairs(nChannel)
        elif revert:
            pairs=getAllPairsUndirected(nChannel)
        else:
            pairs=getAllPairsNoAuto(nChannel)
            
    else:
        #only random pairs 
        pairs = numpy.floor(numpy.random.uniform(0,nChannel,(maxnpairs*1.2, 2)))
        pairs = pairs[(pairs[:,0]!=pairs[:,1]),:]
        pairs = numpy.unique(map(tuple,pairs));

        if pairs.shape[0]<maxnpairs:
            #there might be doubles, only once again to avoid endless loop
            pairs = numpy.concatenate([pairs, numpy.floor(numpy.random.uniform(0, nChannel, (maxnpairs*1.2, 2)))])
            pairs = numpy.unique(map(tuple,pairs));

    return pairs[0:min(pairs.shape[0], maxnpairs),:].astype('int')


def ccf(x, y, axis=None):
    """Computes the cross-correlation function of two series `x` and `y`.
Note that the computations are performed on anomalies (deviations from average).
Returns the values of the cross-correlation at different lags.
Lags are given as [0,1,2,...,n,n-1,n-2,...,-2,-1] (not any more)

:Parameters:
    `x` : 1D MaskedArray Time series.
    `y` : 1D MaskedArray Time series.
    `axis` : integer *[None]*
        Axis along which to compute (0 for rows, 1 for cols). If `None`, the array is flattened first.
    """
    assert(x.ndim == y.ndim, "Inconsistent shape !")
    if axis is None:
        if x.ndim > 1:
            x = x.ravel()
            y = y.ravel()
        npad = x.size + y.size
        xanom = (x - x.mean(axis=None))
        yanom = (y - y.mean(axis=None))
        Fx = numpy.fft.fft(xanom, npad, )
        Fy = numpy.fft.fft(yanom, npad, )
        iFxy = numpy.fft.ifft(Fx.conj()*Fy).real
        varxy = numpy.sqrt(numpy.inner(xanom,xanom) * numpy.inner(yanom,yanom))
    else:
        npad = x.shape[axis] + y.shape[axis]
        if axis == 1:
            if x.shape[0] != y.shape[0]:
                raise ValueError, "Arrays should have the same length!"
            xanom = (x - x.mean(axis=1)[:,None])
            yanom = (y - y.mean(axis=1)[:,None])
            varxy = numpy.sqrt((xanom*xanom).sum(1) * (yanom*yanom).sum(1))[:,None]
        else:
            if x.shape[1] != y.shape[1]:
                raise ValueError, "Arrays should have the same width!"
            xanom = (x - x.mean(axis=0))
            yanom = (y - y.mean(axis=0))
            varxy = numpy.sqrt((xanom*xanom).sum(0) * (yanom*yanom).sum(0))
        Fx = numpy.fft.fft(xanom, npad, axis=axis)
        Fy = numpy.fft.fft(yanom, npad, axis=axis)
        iFxy = numpy.fft.ifft(Fx.conj()*Fy,n=npad,axis=axis).real
    # We juste turn the lags into correct positions:

    iFxy = numpy.concatenate((iFxy[len(iFxy)/2:len(iFxy)],iFxy[0:len(iFxy)/2]))
    return iFxy/varxy


def spikeXCorr(resp, bindt=0.025, maxlag=2., maxnpairs=2e4, startT=-numpy.inf, endT=numpy.inf, channels=None, revert=True):
    # for spikes we can just make a histogram of the pairwise spikes time distances of 2
    # spikes trains and we get the correlation. 

    spikes = getSpikeList(resp, startT, endT, channels=channels)
    nChannel = len(spikes)

    T=resp.Tsim
    if endT < numpy.inf: T=endT
    if startT > 0: T=T-startT
    if maxlag>T: maxlag = T
    
    pairs = getPairs(nChannel, maxnpairs)
    nbins = numpy.floor(T/float(bindt))
    used_bins = numpy.ceil(maxlag/float(bindt))

    length = 2*nbins+1
    used_length = 2*used_bins+1
    avghist = numpy.zeros(used_length)
    edges = numpy.arange(0, T-bindt/2., bindt)
    s=0
    for i in xrange(len(pairs)):
        if pairs[i,0]==pairs[i,1]:  #no auto correlations
            continue
        sp1=spikes[pairs[i,0]]
        sp2=spikes[pairs[i,1]]      
        if (len(sp1)==1 and numpy.isnan(sp1)) or (len(sp2)==1 and numpy.isnan(sp2)):
            continue;
        s += 1
        if len(sp1)>0 and len(sp2)>0:
            hist_1=matplotlib.mlab.hist(sp1, edges)
            hist_2=matplotlib.mlab.hist(sp2, edges)
            h = ccf(hist_1[0],hist_2[0])
            h_used = h[numpy.floor(length-used_length)/2 :-numpy.floor(length-used_length)/2+1]
            avghist += h_used
            if revert:
                avghist += h_used[::-1]
                s+=1

    outdat = avghist/s;

    return dict(data=outdat, lag=bindt*numpy.arange(-len(avghist)/2+1, len(avghist)/2+1))

def spikeXCorrDist(resp, bindt=0.003, maxlag=0.1, maxnpairs=2e4, startT=-numpy.inf, endT=numpy.inf, channels=None,
                   spatialbinsize=1., maxdist=10., toroid=True):
#                   spatialbinsize=0.5e-3, maxdist=3e-3, toroid=True):
    
    if len(resp.pos_x) != len(resp.pos_y):
        raise exception('len of pos vectors differen')
    
    pos = numpy.array([[resp.pos_x[i], resp.pos_y[i]] for i in xrange(len(resp.pos_x))])
    
    if channels:
        pos=pos[channels]
        
    bounds1=pos.min(0)
    bounds2=pos.max(0)
    ndist = int(numpy.ceil(maxdist/spatialbinsize)+1)
    
    spikes = getSpikeList(resp, startT, endT, channels=channels)
    nChannel = len(spikes)

    T=resp.Tsim
    if endT < numpy.inf: T=endT
    if startT > 0: T=T-startT
    if maxlag>T: maxlag = T
    
    pairs = getPairs(nChannel, maxnpairs)
    nbins = numpy.floor(T/float(bindt))
    used_bins = numpy.ceil(maxlag/float(bindt))

    length = 2*nbins+1
    used_length = 2*used_bins+1
    avghist = numpy.zeros((used_length,ndist))
    edges = numpy.arange(0, T-bindt/2., bindt)
    s = numpy.zeros(ndist)
    for i in xrange(len(pairs)):
        if pairs[i,0]==pairs[i,1]:  #no auto correlations
            continue
        sp1=spikes[pairs[i,0]]
        sp2=spikes[pairs[i,1]]
        if toroid:
            dd1 = abs(pos[pairs[i,0],:] - pos[pairs[i,1],:]) 
            dd2 = abs(pos[pairs[i,0],:] - bounds1) + abs(bounds2 - pos[pairs[i,1],:])
            dd3 = abs(pos[pairs[i,0],:] - bounds2) + abs(bounds1 - pos[pairs[i,1],:])
            dd = numpy.array([dd1,dd2,dd3]).min(0)
        else:
            dd = abs(pos[pairs[i,0],:] - pos[pairs[i,1],:])
        dist = numpy.sqrt((dd**2).sum())
        idd = min(numpy.floor(dist/spatialbinsize), ndist-1);

        if (len(sp1)==1 and numpy.isnan(sp1)) or (len(sp2)==1 and numpy.isnan(sp2)):
            continue
        
        s[idd] += 1
        if len(sp1)>0 and len(sp2)>0:
            hist_1=matplotlib.mlab.hist(sp1, edges)
            hist_2=matplotlib.mlab.hist(sp2, edges)
            h = ccf(hist_1[0],hist_2[0])
            avghist[:,idd] += h[numpy.floor(length-used_length)/2 :-numpy.floor(length-used_length)/2+1]

    for i in range(ndist):
        avghist[:,i] /=s[i]
        
    outdat = avghist

    return dict(data=outdat, lag=bindt*numpy.arange(-len(avghist)/2+1, len(avghist)/2+1),
                             spatial=numpy.arange(float(spatialbinsize), float(maxdist)+spatialbinsize*3./2., float(spatialbinsize)))


def spikeXCorr_old(resp, bindt=0.025, maxlag=2., maxnpairs=2e4, startT=-numpy.inf, endT=numpy.inf, channels=None):
    # for spikes we can just make a histogram of the pairwise spikes time distances of 2
    # spikes trains and we get the correlation. 

    spikes = getSpikeList(resp, startT, endT, channels=channels)
    nChannel = len(spikes)
    T=resp.Tsim
    if endT < numpy.inf: T=endT
    if startT > 0: T=T-startT
    if maxlag>T: maxlag = T

    edges = numpy.linspace(0, maxlag, numpy.ceil(maxlag/bindt))
    edges = numpy.concatenate([-edges[::-1][:-1],edges])

    n = numpy.ceil(T/bindt)
    scale = numpy.arange(n, n-numpy.ceil(maxlag/bindt), -1)
    scale = numpy.concatenate([scale[::-1][:-1],scale])
    scale2 = numpy.arange(n, 1, -1)
    scale2 = numpy.concatenate([scale2[::-1][:-1],scale2])
    scale /= scale2.sum()

    pairs = getPairs(nChannel, maxnpairs)
    avghist = numpy.zeros(edges.shape)
    s=0; N1=0; N2=0
    for i in xrange(len(pairs)):
        if pairs[i,0]==pairs[i,1]:  #no auto correlations
            continue
        sp1=spikes[pairs[i,0]]
        sp2=spikes[pairs[i,1]]
        if (len(sp1)==1 and numpy.isnan(sp1)) or (len(sp2)==1 and numpy.isnan(sp2)):
            continue;
        s += 1
        N1 += len(sp1)
        N2 += len(sp2)

        if len(sp1)>0 and len(sp2)>0:
            d = spktdist(sp1, sp2, maxlag)
            if len(d)>0:
                Nm = matplotlib.mlab.hist(d, bins=edges)[0]               
                h = Nm
#                h = Nm/scale/T**2 - len(sp1)*len(sp2)*bindt/T
#                h = Nm - len(d)*bindt/T
#                h = Nm/scale - len(sp1)*len(sp2)*bindt/T
                avghist = avghist + h

    outdat = avghist/scale/N1/N2*s;

    return dict(data=outdat, lag=edges)


def channelCorr(resp, bin=0.01, startT=0.0, endT=1.0, max_lag=2.0, sigma=0.01, 
                only_auto=False, exclude_auto=True):
    '''
    '''

    spikes = getSpikeList(resp)
    nChannels = len(resp.channel)
    N = numpy.ceil((endT-startT)/bin)
    if only_auto:
       m = nChannels
       exclude_auto = False
    elif exclude_auto:
       m = int((nChannels^2-nChannels)/2)
    else:
       m = int((nChannels^2-nChannels)/2) + nChannels

    lag = numpy.floor(max_lag/bin)
    dat = []
    s=-1
    for i in range(nChannels):
        spikes_i = numpy.floor((spikes[i] - startT)/bin)
        if len(spikes_i) == 0:
            continue
        x = subSDF(spikes_i, N, bin, sigma=sigma)
        for j in range(i+1-int(exclude_auto)):
            if i != j:
                if only_auto:
                     continue
                spikes_j = numpy.floor((spikes[j] - startT)/bin)
                if len(spikes_j) == 0:
                    continue
                y = subSDF(spikes_j, N, bin, sigma=sigma)
                c = scipy.signal.correlate(x, y)
                s = s+1
            else:
                s = s+1;
                c = scipy.signal.correlate(x, x)
            c = c[N-lag-1:N+lag]
            dat.append(c)
    
    return dict(data=dat, lag=numpy.arange(-max_lag, max_lag+bin, bin))


def lfpPSD(resp, dt=1e-4, tau=0.01, startT=0.0, endT=numpy.inf, NFFT=256, detrend=matplotlib.mlab.detrend_none, noverlap=0):
    ''' computes the PSD of the LFP given the spiketimes
    (see C. Bedard, H. Kroeger, and A. Destexhe 2006)'''
    
    T=endT-startT
    if numpy.isnan(T) or numpy.isinf(T) or numpy.isneginf(T):
        T=None

    spiketimes=getCombindedSpikeList(resp, startT=startT, endT=endT)
        
    C = createC(spiketimes, dt, tau, T)
    
    (power, freq) = matplotlib.mlab.psd(C, NFFT=NFFT, Fs=1/dt, detrend=detrend, noverlap=noverlap) 
    
    return dict(data=power, freq=freq)


def createC(spiketimes, dt, tau=0.01, T=None):
    ''' computes the driving currents of LFP '''
    
    if T is None or numpy.isnan(T) or numpy.isinf(T) or numpy.isneginf(T):
        T=numpy.ceil(spiketimes.max())            # TODO better ;)
    
    D=numpy.zeros(T/dt)
    for spike in spiketimes:
        D[int(spike/dt)] = 1
        
    n = numpy.arange(0, numpy.ceil(5*tau/dt))
    h = numpy.exp(-n*dt/tau)
    C = numpy.convolve(D, h, mode='same')     
    
    return C


if __name__ == '__main__':
    startT = 0.0
    endT = 2.0
    Tsim = 2.0
    freq = 20.0
    
    dt = 0.001
    nSpikes = 40
    
    spikes = numpy.random.uniform(startT, endT, nSpikes)
    spikes.sort()
    
    sdf = getSDF(spikes, dt, startT, endT)
    sdfg = getSDF(spikes, dt, startT, endT, sigma=0.01)
    
    resp = dataformats.Response(Tsim=Tsim)
    for i in range(3):
        sp=numpy.random.uniform(0, Tsim, freq*Tsim)
        sp.sort()
        resp.channel.append(dataformats.Channel(sp))
    
    resp.channel.append(dataformats.Channel())

    rates = firingRate(resp, bin=0.2, startT=0.0, endT=1.0)
    
    rates2 = firingRate(resp, bin=0.2, startT=0.0, endT=1.0, dt=0.05)
    
    isi = isiDist(resp, bin=0.2, startT=0.0, endT=1.0)
    isi2 = isiDist(resp, bin=0.2, startT=0.0, endT=1.0, dt=0.05)

    isicv = isiCV(resp, bin=0.2, startT=0.0, endT=1.0)
    isicv2 = isiCV(resp, bin=0.2, startT=0.0, endT=1.0, dt=0.05)
    
    fano = fanoFactorTime(resp, bin=0.2, startT=0.0, endT=1.0)
    fanofact = fanoFactor(resp)
    meanfanofact = meanFanoFactor(resp)
    
#    bursts = calcBursts(spikes, 0.01, 3)
    bursts=burstRate(resp, minBurst=2, maxBurst=20)
    
    chcorr=channelCorr(resp, bin=0.001, startT=0.0, endT=1.0, max_lag=2.0, sigma=0.005, only_auto=False, exclude_auto=True)

    max_lag=0.05
    bin=0.001
    
    chcorr=channelCorr(resp, bin=bin, startT=0.0, endT=1.0, max_lag=max_lag, sigma=0.005, exclude_auto=False)
    tidx = numpy.arange(-max_lag, max_lag+bin, bin)
    
    corr_def = dict(max_lag=0.05, bin=0.001, sigma=0.005, exclude_auto=True)
    chcorr2=channelCorr(resp, startT=0.0, endT=1.0, **corr_def)
    
    lfp=lfpPSD(resp, dt=1e-4, tau=0.01, startT=0.0, endT=1.0, NFFT=256)
    
#    chcorr=channelCorr(resp, bin=0.001, startT=0.0, endT=1.0, max_lag=0.5, sigma=0.005, exclude_auto=True)
    
    ele = getEleSpikeList(resp,1,maxele=100)

    if 0:
        import dataformats
        sim = '/home/workspace/malte/saves/pysim/sim_WscaleAndLRW_ephysBackgroundSpont_17x17/WscaleAndLRW_ephysBackgroundSpont_17x17.260'

        resp = dataformats.Response()
        resp.load(sim,'/Response');

        ele_lst = getEleSpikeListDist(resp,3,nele=10,forcepos='2/3',zsigmafrac=0.1) # force in layer 2/3
