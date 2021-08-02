import logging
import numpy
#import time

from numpy import array, matrix, nan, unique, multiply, sum, prod, dot, sqrt, log, zeros, ones, atleast_1d




class ReadoutException(Exception): 
    pass


class Readout(object):
    def __init__(self):
        self.trained=False

    def train(self, trainSet_X, trainSet_Y):
        pass

    def apply(self, X):
        pass

    def analyse(self, set_X, set_Y):
        pass


def unique_index(l):
    u = unique(l).tolist()
    index_l = [u.index(x) for x in l]
    
    return (u, index_l)


def corr_coeff(x, y):
    x = numpy.asarray(x)
    y = numpy.asarray(y)

#    mx = x.mean()
#    my = y.mean()
#    coeff = numpy.dot((x-mx),(y-my)) / (numpy.sqrt(numpy.dot((x-mx),(x-mx))*numpy.dot((y-my),(y-my))))
    coeff = numpy.corrcoef(x,y)[0,1]

    return coeff


def confusion_matrix(o, y, values):
    n = len(values)
    C = numpy.array(zeros((n,n)))

    o = numpy.asarray(o)
    y = numpy.asarray(y)

    for i in range(n):
        oo = o[(y==values[i])]
        if len(oo)>0:
            for j in range(n):
                C[i,j] = numpy.sum((oo == values[j]).astype(numpy.int32))

    return C


def mi_from_count(argNXY):
    ''' MI_FROM_COUNT compute mutual information from joint count table

    Syntax
   
      (MI,HX,HY,HXY) = mi_from_count(Nxy)

    Arguments
     
      Nxy ... joint count table; i.e. Nxy(i,j) is the number of occurrences
             (or probabilty) of observation of pairs (i,j).

       MI ... mutual information
       HX ... entropy of X
       HY ... entropy of Y
      HXY ... joint entropy
 
    Description
    
      [MI,HX,HY,HXY] = mi_from_count(Nxy) computes the mutual
      information and the related entropies from the given joint cout
      table (joint probability density function).
  
    Algorithm
 
      MI = SUM_i SUM_j Nxy(i,j) * ld Nxy(i,j) / ( nx(i) * ny(j) )
      nx(i) = SUM_j Nxy(i,j)
      ny(j) = SUM_i Nxy(i,j)'''

    MI=nan; Hx=nan; Hy=nan; Hxy=nan

    argNXY=numpy.array(argNXY)

    if prod(argNXY.shape) == 0:
        return (MI, Hx, Hy, Hxy)

    pxy = argNXY/argNXY.sum()

    logging.debug("pxy"+ str(pxy))
    
#    pxy[:, all(pxy==0, 0)] = []
#    pxy[all(pxy==0, 1), :] = [];
    
    px = pxy.sum(1)
    py = pxy.sum(0)
    
    Hx = -sum(multiply(px, log(px)))/log(2)
    Hy = -sum(multiply(py, log(py)))/log(2)
    
    p=pxy/(px*ones((1,pxy.shape[1])))
    
    p[p==0]=1
    H=-sum(multiply(p,log(p)),1)/log(2)
    Hyx=sum(multiply(px,H))
    
    MI = Hy - Hyx
    Hxy = Hx + Hyx
    
    return (MI, Hx, Hy, Hxy)



if __name__=='__main__':
    O = array([0,1,1,2,0,1])
    y = array([0,1,2,0,0,1])
    uy = array([0,1,2])

    O = array([0,1,1,2,0,1])
    y = array([0,1,1,2,0,1])
    uy = array([0,1,2])

    O = [0,1,1,2,0,1]
    y = [0,1,2,0,0,1]
    uy = [0,1,2]

    cm = confusion_matrix(O, y, uy)
    
    (mi, hx, hy, hxy) = mi_from_count(cm) 
