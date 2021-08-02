"""
functions to define movie stimlui 
typically used to stimulate the Retina and LGN model

MJR, 2/2008
"""

import scipy
import scipy.io

import numpy
import sys
import types

from pyV1.utils import default


def getRFMovieStm(dims,movtregion,sesName='d04nm1',grpName='movie1',chan=1,movdir='',col='bw',fixedRFsize=None,scale = False):
    """extracts a fragment of the RF movie of a certain channel and
    session. movtregion is from beginning to end of movie"""

    if not len(movdir):
        movdir = default.dirs['movies']
    
    if fixedRFsize is None:
        # get the file path
        fname = '%s/RF%s_%s_ch%02d.mat' % (movdir,grpName.lower(),sesName.lower(),chan)
    else:
        fname = '%s/RF%s_%s_ch%02d_FIXED%1.2f.mat' % (movdir,grpName.lower(),sesName.lower(),chan,fixedRFsize)

    mov = scipy.io.loadmat(fname)['RFmovie']
    mov = mov.astype(numpy.float)/255.0;
    
    if len(mov.shape)==4:
        #color available
        if col.lower() == 'red':
            mov = mov[:,:,:,0]
        elif col.lower() == 'green':
            mov = mov[:,:,:,1]
        elif col.lower() == 'blue':
            mov = mov[:,:,:,2]
        elif col.lower() == 'rg':
            #mov = (mov[:,:,:,0]-mov[:,:,:,1])/(mov[:,:,:,0]+mov[:,:,:,1]) # its color contrast from -1..1 ()
            mov = (mov[:,:,:,0]-mov[:,:,:,1])/2.0 + 0.5 # its from 0..1 ()

        elif col.lower() == 'bw':
            mov = numpy.mean(mov,3)
        else:
            raise Exception(" dont know color")

    if sesName.lower()=='a98nm5':
        dt = 0.049584 #fixed to frame, some frames may be dropped however
        degperpx = numpy.array([10.0/320.0,7.5/240.0])
        fovea = numpy.array((+2.0,-5.82))            # and center of stim  (according to yusuke/panzeri)

    elif sesName.lower()=='d04nm1':
        dt = 0.033404723627685;
        degperpx = numpy.array([10.0/320.0,7.5/240.0])
        fovea = numpy.array((+1.0,-3.0))
    else:
        raise("dont know frame dt and stimulus sizes (check experiments)")

    #load measured RF size data
    elerfsizes = scipy.io.loadmat('%s/RFsizes.mat' % (movdir))['rf']
    
    rf = elerfsizes.__dict__[sesName.lower()][chan-1]
    eccentricity = numpy.sqrt(numpy.sum((rf.center - fovea)**2))

    if fixedRFsize!=None:
        vf = numpy.array([fixedRFsize,fixedRFsize]).astype(numpy.float)
    else:
        #to RFmovie size
        vf = rf.size
    
    ireg = numpy.floor(numpy.array(movtregion)/dt).astype(numpy.int)
    
    if ireg[0]<0 or ireg[0]>mov.shape[2]:
        raise "invalid movie region start time given"

    if ireg[1]<0 or ireg[1]>mov.shape[2]:
        raise "invalid movie region end time given"

    mov = mov[::-1,:,ireg[0]:ireg[1]]

    if not dims is None: #otherwise keep dimensions (LGN will do resize)
        dims = numpy.array(dims)
   
        if not (dims[0]==mov.shape[0] and dims[1]==mov.shape[1]):

            for ii in range(2):
                degperpx[ii] *=  float(mov.shape[ii])/float(dims[ii])

            d = numpy.r_[dims,mov.shape[2]]
            X = numpy.zeros(d).astype(numpy.int)

            for t in range(mov.shape[2]):
                im=scipy.misc.toimage(mov[:,:, t],mode='I')
                im=im.resize(dims[0:2])
                X[:,:,t] =scipy.misc.fromimage(im) 
                mov = X

    if scale:
        mov -= mov.min()
        mov /= mov.max()

    #info about movie
    movinfo = {}
    movinfo['tstart'] = movtregion[0]
    movinfo['movtregion'] = movtregion
    movinfo['chan'] = chan
    movinfo['grpName'] = grpName 
    movinfo['sesName'] = sesName
    movinfo['col'] = col
    movinfo['fixedRFsize'] = fixedRFsize
    movinfo['dt'] = dt
    movinfo['n'] = (mov.shape[0],mov.shape[1])
    movinfo['degperpx'] = degperpx
    movinfo['rfsize'] = rf.size
    movinfo['rfcenter'] = rf.center
    movinfo['fovea'] = fovea
    movinfo['eccentricity'] = eccentricity
    movinfo['visualfield'] = vf
    movinfo['scale'] = scale

    return (mov, movinfo)
    


def plotMovieFrames(mov,npic=100,cmap=None,dt=0.049584,tstart=0):
    """ plots n lin-spaced frames of movie"""     
    import pylab
    import utils.mscfuncs as msc

    pylab.figure()

    movtregion = [tstart,tstart+mov.shape[2]*dt]

    npic = 100
    r = numpy.int(numpy.ceil(npic/numpy.sqrt(npic)))
    s = 0

    for i in numpy.linspace(0,mov.shape[2]-1,npic):
        ii = numpy.floor(i)
        s = s+1
        
        pylab.subplot(r,r,s)
        msc.imagesc(mov[:,:,i],scale=[0,1],cmap=cmap)
        pylab.title('%1.2f sec' % (ii*dt+movtregion[0]),fontsize=6 )
        pylab.axis('off')


if __name__ == '__main__':

    dims = numpy.array([50, 50])

    sesName='a98nm5'
    grpName='movie'
    chan=1
    movtregion = [10.0, 40.0]
    dt = 0.049584

    mov = getRFMovieStm(dims,movtregion,sesName,grpName,chan)

    pylab.colormaps()

    plotMovieFrames(mov,npic=100,cmap=pylab.cm.gray,tstart=movtregion[0])
    
    dim_t=300
    Tsim=1.2
    
    nTimeSteps = 300
    stmframes = numpy.r_[numpy.zeros(125), 
                         numpy.ones(25), numpy.zeros(25), 
                         numpy.ones(25), numpy.zeros(25), 
                         numpy.ones(25), numpy.zeros(50)]
    letters = ['A','B','E']
    
    dims = numpy.r_[dim, dim_t]
    stim = generateLetterStim(dims, stmframes, letters)

    pylab.show()
