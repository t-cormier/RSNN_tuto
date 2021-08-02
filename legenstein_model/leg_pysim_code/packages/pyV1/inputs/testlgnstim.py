'''
some tests of the simple LGN

    Klaus Schuch   schuch@igi.tugraz.at
    
    March 2007
'''


from bitmapstim import *
from lgn import *

import numpy
import pylab


def concBitmap(s, ind):
    emptycol = numpy.zeros([s.shape[0], 2])
    
    V = emptycol
    for t in ind:
        V = numpy.hstack((V, s[:,:,t]))
        V = numpy.hstack((V, emptycol))
        
    return V


def plotResults(lgn, stim, ind, s=None):
    pylab.figure()
    
    V = concBitmap(stim, ind)
    pylab.subplot(5,1,1)
    pylab.imshow(V, cmap=pylab.cm.gray)
    pylab.title('stimulus')
    pylab.xticks([]); pylab.yticks([])
    
#    V = concBitmap(lgn.ret.S, ind)
#    pylab.subplot(5,1,2)
#    pylab.imshow(V)
#    pylab.title('retina output')
#    pylab.xticks([]); pylab.yticks([])
#    pylab.colorbar()

#    tit = ['nonlagged on cells', 'lagged on cells', 'nonlagged off cells', 'lagged off cells']
    tit = ['nonlagged on cells', 'nonlagged off cells', 'lagged on cells', 'lagged off cells']
    data=lgn.analogLGNoutput.reshape(lgn.shape, order='C')
    for k in range(len(tit)):
        V = concBitmap(data[k,:,:,:], ind)
#q        V = concBitmap(lgn.analogLGNoutput[k], ind)
        pylab.subplot(5,1, 2+k)
        pylab.imshow(V)
        pylab.title(tit[k])
        pylab.xticks([]); pylab.yticks([])
#        pylab.colorbar()

#    if s is None:
#        s = lgn.generate()
    
    ##plot the stimulus
    #pylab.figure()
    #s.plot()
    #pylab.title('LGN spiking activity')
    #pylab.xlabel('time'); pylab.ylabel('nrn index')
    
    ## plot the frequencies
    #pylab.figure()
    #pylab.imshow(lgn.gamma.f, aspect='auto', origin='lower')
    #s = [str(k) for k in numpy.linspace(0.0, lgn.Tstim, 11)]
    #pylab.xticks(numpy.linspace(0, lgn.gamma.f.shape[1]-1, 11), s)
    #pylab.title('frequency of the neuron')
    #pylab.xlabel('time'); pylab.ylabel('nrn index')
    #pylab.colorbar()
    
    
if __name__ == '__main__':    
    
## stim generation test
#    nTimeSteps = 10
#    dims = numpy.array([50, 50, nTimeSteps])      # dimt, dimx,dimy vector as in LGN object    
#    stmframes = [0, 1, 1, 0, 1, 1, 0, 1, 1, 0]    # one for stm zero for gap frame (for each DIM_T one entry!)
#    letters = ['A','B','E']
#
#    stim = generateLetterStim(dims, stmframes, letters)
#    
#    pylab.figure()

#    V = concBitmap(stim, range(nTimeSteps))    
#    pylab.subplot(2,1,1)
#    pylab.imshow(V, cmap=pylab.cm.gray)
#    pylab.xticks([]); pylab.yticks([])
#    pylab.colorbar()
#

## retina Test
#    ret = Retina()    
#    (son, soff) = ret.processStimulus(stim)
#
#    V = concBitmap(ret.S, range(nTimeSteps))
#    pylab.subplot(2,1,2)
#    pylab.imshow(V)
#    pylab.xticks([]); pylab.yticks([])
#    pylab.colorbar()



# LGN Tests
    dim=numpy.array([50, 50])

# Example with letter stimulus        
#    stmframes = numpy.r_[numpy.zeros(5), 
#                         numpy.ones(25), numpy.zeros(25), 
#                         numpy.ones(25), numpy.zeros(25), 
#                         numpy.ones(25), numpy.zeros(5)]
    
    stmframes = numpy.r_[numpy.zeros(1), numpy.ones(40), numpy.zeros(19)]
    
    Tstim=1.0
#    Tstim=0.2
    nTimeSteps = len(stmframes)
    
#    letters = ['A','B','E']
    letters=['A']
    dims = numpy.r_[dim, nTimeSteps]
    stim = generateLetterStim(dims, stmframes, letters)
    
    lgn = LGN(dt=Tstim/60.0)
    lgn.processStimulus(stim, Tstim=Tstim)
    
    ind = numpy.arange(0, 60, 2)
    plotResults(lgn, stim, ind)
    
    pylab.xticks([pylab.xlim()[0], pylab.xlim()[1]/2.0, pylab.xlim()[1]], ['0.0', str(Tstim/2.0), str(Tstim)+' s'])
# Example with spot stimulus
#===============================================================================
#    Tsim=0.2xxx
#    stmframes = numpy.r_[numpy.zeros(1), numpy.ones(15), numpy.zeros(39)]
#    nTimeSteps = len(stmframes)
#    
#    ret = Retina()
#    spot = numpy.ones(ret.k.shape)*(ret.k>0)
#    
#    nBitmaps = 1
#    bitmaps = numpy.zeros((ret.k.shape[0], ret.k.shape[1], nBitmaps))
#    bitmaps[:,:,0]=spot
#    
#    stim = generateBitmapStim(bitmaps, stmframes)
#    
#    lgn = LGN(Tsim=Tsim)
#    lgn.processStimulus(stim)
#    
#    ind = numpy.arange(0, 40, 1)
#    plotResults(lgn, stim, ind)
#===============================================================================
    
    pylab.show()
        
