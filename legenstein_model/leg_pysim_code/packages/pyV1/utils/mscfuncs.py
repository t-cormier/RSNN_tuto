
import numpy

def imagesc(X,scale=None,cmap=None,**kwds):
    """ implements imagesc as in matlab"""
    import pylab
    
    #pylab.figure()

    if scale is None:
        vmin = None
        vmax = None
    else:
        vmin = scale[0]
        vmax = scale[1]

    if not kwds.has_key('extent'):
        kwds['extent']=[0,1,0,1]

    if not kwds.has_key('interpolation'):
        kwds['interpolation']='nearest'

    a = pylab.imshow(X,vmin=vmin,vmax=vmax,origin='lower',cmap=cmap,**kwds)
    return a

def xlimall(fig,lim):
    """ sets xlim for all axes of figure FIG to LIM """
    import pylab

    if fig is None:
        fig = pylab.gcf()

    ax_lst = fig.get_axes()

    for ax in ax_lst:
        ax.set_xlim(lim)


def ylimall(fig,lim):
    """ sets ylim for all axes of figure FIG to LIM """
    import pylab
    if fig is None:
        fig = pylab.gcf()

    ax_lst = fig.get_axes()

    for ax in ax_lst:
        ax.set_ylim(lim)


def setall(fig,field,value):
    """ sets a fields for all axes of figure FIG to LIM """
    import pylab

    if fig is None:
        fig = pylab.gcf()

    ax_lst = fig.get_axes()

    for ax in ax_lst:
        pylab.set(ax,field,value)


def i2s(sz,ind):
    """works like ind2sub but with v = s2i(sz,ind) instead of
    [v1,v2,v3,...] = ind2sub(sz,ind)
    works along dimension 2 (i.e. rows specify the index)
    CAUTION no error checking (if index out of range)"""

    ind = numpy.array(ind) + 1
    sz = numpy.array(sz)
    sub=numpy.zeros([ind.size,sz.size]).astype(numpy.int);

    for i in range(len(sz)-1):

        sub[:,i] = numpy.mod(ind,sz[i]);
        sub[sub[:,i]==0,i] = sz[i];
        ind = (ind - sub[:,i])/sz[i] + 1;
  
    sub[:,i+1] = ind;
    sub = sub - 1

    return sub



def s2i(sz,v):
    """ works like sub2ind but with s2i(sz,[v1,v2,v3,...]) instead of
    sub2ind(sz,v1,v2,v3,...)
    works along dimension 2 (i.e. rows specify the indices)
    CAUTION no error checking (if index out of range)"""
    
    index = numpy.array(v[:,0]).astype(numpy.int)
    sc=1
    for i in range(1,len(sz)):
        sc = sc*(sz[i-1]);
        index = index + sc*(v[:,i]);
        
    return index
  
   

  
  
if __name__=='__main__':

    g1 = [0,2,4,15,119,500]
    g = i2s([3,4,2,5],g1)
    
    g2 = s2i([3,4,2,5],g)

    print g1
    print g
    print g2
