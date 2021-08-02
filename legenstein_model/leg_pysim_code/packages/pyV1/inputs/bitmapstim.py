"""
some function to define Bitmap stimuli
typically used to stimulate the Retina and LGN model

    Klaus Schuch   schuch@igi.tugraz.at
    
    March 2007
"""

import scipy
import scipy.signal
#import scipy.misc

import numpy
import sys
import types

from os import path


# find the path to the sample bitmaps

__stimulipath__ = path.dirname(__file__)

# if you use it with ipython, this file must be in the working directory 
if path.basename(__file__)[0:10] == 'FakeModule':        
    __stimulipath__ = ''
    
__stimulipath__ = path.join(__stimulipath__, 'stimuli')



def gratings(dims, freq, angle, phi):
    '''  generates a grating picture'''
    
    dt = 1.0/dims.mean()
    X, Y = numpy.meshgrid(numpy.arange(0, 1, dt), arange(0, 1, dt))
    out = numpy.sin(2*numpy.pi*freq*(numpy.cos(angle)*X + numpy.sin(angle)*Y) + phi)
    
    return out


def generateGratingStim(dims=None, v=0.4, freq=2, theta=0):
    '''generates moving gratings stimuli'''
    
    if dims is None:
        dims = numpy.array([12, 15, 15])                  # dimt, dimx,dimy vector as in LGN object    

    nTimeSteps = dims[2]    
    stim = numpy.zeros((dims[0], dims[1], nTimeSteps))

    for t in range(nTimeSteps):
        stim[:,:,t] = gratings(dims, freq, theta, v*2*numpy.pi*t/nTimeSteps)
    
    return stim    


def generateBitmapStim(bitmaps, stmframes=None, dims=None):
    '''  generates bitmaps stimuli
  bitmaps   ..,
  dims      ... dimension of the bitmap [xdim, ydim]
  stmframes ... 
    '''
    
    if stmframes is None:
        # one for stm zero for gap frame (one entry for each timestep!)
        stmframes = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0]  
    
    nTimeSteps = len(stmframes)
    
    if dims is None:
        (dimx, dimy) = bitmaps.shape[0:2]
        dims = numpy.array([dimx, dimy, nTimeSteps])      # dimt, dimx, dimy vector as in LGN object    

        
    
    if nTimeSteps <> dims[2]:
        #interpolation
        n = numpy.ceil(dims[2]/nTimeSteps)
        stmframes = numpy.tile(stmframes,[n,1]).flatten(1)
        dims[2] = len(stmframes)                     # make bigger if necessary!!
        nTimeSteps = len(stmframes)
        
    if max(stmframes) > bitmaps.shape[2]:
        raise Exception('requested Bitmap is not defined!!')
        
    X = numpy.zeros(dims)
    for t in range(nTimeSteps):
        if stmframes[t] > 0:
            im=scipy.misc.toimage(bitmaps[:,:, stmframes[t]-1],mode='F')
            im=im.resize(dims[0:2])
            X[:,:,t] =scipy.misc.fromimage(im) 
            
    return X


def generateLetterStim(dims=None, stmframes=None, letters=None):
    ''' generates letter (or other) stimuli as in Dankos stim
  paradigma dims ... dimension of the bitmap [xdim, ydim] stmframes
  ...  letters ...
   '''
    
    if stmframes is None:
        # 1 for stm, 0 for gap frame (one entry for each timestep!)
        stmframes = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0]  
    
    nTimeSteps = len(stmframes)
    
    if dims is None:
        dims = numpy.array([15, 15, nTimeSteps])      # dimt, dimx, dimy vector as in LGN object    
    
    if letters is None:
        letters = ['x', '+', 'x']
    
    if nTimeSteps <> dims[2]:
        raise Exception('dim_t - stmframes mismatch!!')
    
    # compose stimulus
    
    X = numpy.zeros(dims)
    l = -1
    for t in range(nTimeSteps):
        if stmframes[t]:
            if t>0:
                if stmframes[t-1] == 0:
                    l+=1
            else:
                l+=1
        
        if l >= len(letters):
            break
        
        if stmframes[t]:
            X[:,:,t] = makeBitmap(dims[0:2], letters[l])
    
    return X


def gaborFilter(dim, theta, gsize=3, lambd=3, gamma=1):
    '''  generates a gabor filter
  dim   ... x, y dimension of the filter
  theta ...
  gsize ...
  lambd ...
  gamma ...
    '''
    
    # gamma below one means elongation, above sagital shrinking
    sigma=1.0
    phi=0.0

    if numpy.isscalar(dim):
        dim = numpy.array([dim, dim])
        
    elif type(dim) != type('numpy.array'):
        dim = numpy.array(dim)
    
    x = numpy.atleast_2d(numpy.linspace(-gsize*sigma, gsize*sigma, dim[0]))
    y = numpy.atleast_2d(numpy.linspace(-gsize*sigma, gsize*sigma, dim[1]))
    
    x = numpy.kron(numpy.ones((dim[1], 1)), x).T
    y = numpy.kron(numpy.ones((dim[0], 1)), y)
    
    xx =   x * numpy.cos(theta) + y*numpy.sin(theta)
    yy = - x * numpy.sin(theta) + y*numpy.cos(theta)
    
    GabF = numpy.exp(-(xx**2 + gamma**2 * yy**2)/(2 * sigma**2)) * numpy.cos(2*numpy.pi*xx/lambd+phi)
    
    # approx. (sic!) center is reported 
    # center = ceil(dim/2)
    
    return GabF


def makeBitmap(dims, sym):
    '''  create the bitmap specified by sym
   dims ... dimension of the bitmap [xdim, ydim]
   sym  ... symbol of the bitmap 
            (+, -, |, x, =, ||, //, \\) created with GaborFilter
            A, B, C, D, E created from the bmp file 
   '''
   
    foreground = 1.0
    background = 0.0
    
    bitmap = numpy.ones(dims)*background
    
    # for + - x |
    gsize1 = 6; wavelen1 = 10; shrink1 = 0.2
    # for || = \\ //
    gsize2 = 4; wavelen2 = 2.5; shrink2 = 0.2
    
    if sym in ('plus', '+'):
        bitmap = gaborFilter(dims, numpy.pi/2, gsize1, wavelen1, shrink1)/2.0
        bitmap = bitmap + gaborFilter(dims, 0, gsize1, wavelen1, shrink1)/2.0
    elif sym in ('minus', '-'):
        bitmap = gaborFilter(dims, 0, gsize1, wavelen1, shrink1)
    elif sym in ('pipe', '|'):
        bitmap = gaborFilter(dims, numpy.pi/2, gsize1, wavelen1, shrink1)
    elif sym in ('times', 'x'):
        bitmap = gaborFilter(dims, numpy.pi/4, gsize1, wavelen1, shrink1)/2.0
        bitmap = bitmap + gaborFilter(dims,-numpy.pi/4, gsize1, wavelen1, shrink1)/2.0
    elif sym == '=':
        bitmap =  - gaborFilter(dims, 0, gsize2, wavelen2, shrink2)
    elif sym == '||':
        bitmap =  - gaborFilter(dims, numpy.pi/2, gsize2, wavelen2, shrink2)
    elif sym == r'//':
        bitmap =  - gaborFilter(dims, numpy.pi/4, gsize2, wavelen2, shrink2)
    elif sym == r'\\':
        bitmap =  - gaborFilter(dims, -numpy.pi/4, gsize2, wavelen2, shrink2)

    bitmap = (bitmap*(bitmap>0))/2*foreground;

    if sym in ('A','B','C','D','E'):
        if sym == 'A':   sf = 'Alp99.bmp'
        elif sym == 'B': sf = 'Alp100.bmp'
        elif sym == 'C': sf = 'Alp101.bmp'
        elif sym == 'D': sf = 'Alp102.bmp'
        elif sym == 'E': sf = 'Alp103.bmp'
        
        fname = path.join(__stimulipath__, sf)
        bitmap = scipy.misc.imread(fname,1)

#        bitmap = pylab.imread(fname)             # only for png

        bitmap = 1.0*scipy.misc.imresize(bitmap, dims)
   
        # scale between 0.0 and 1.0
        bitmap /= 255.0    
    
    
    return bitmap



if __name__ == '__main__':
    import pylab
    dim = numpy.array([50, 50])

    bmp = makeBitmap(dim, 'A')
    pylab.figure()
    pylab.imshow(bmp)
    pylab.colorbar()
    
    bmp = makeBitmap(dim, '//')
    pylab.figure()
    pylab.imshow(bmp)
    pylab.colorbar()

    bmp = makeBitmap(dim, r'\\')
    pylab.figure()
    pylab.imshow(bmp)
    pylab.colorbar()
        
    
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
