import scipy.linalg
import utils
import time
import numpy
import copy

from readout import *
from lsqnonneg import lsqnonneg


class LinearRegression(Readout):
    ''' Linear Regression based on LMS'''
    def __init__(self, range=(-1,1), addBias=True):
        super(LinearRegression, self).__init__()
        self.time = -1
        self.range = range
        self.addBias = addBias


    def train(self, trainSet_X, trainSet_Y):
        '''train linear regression'''
        t0 = time.time()

        # onvert to array
#        if type(trainSet_X) is not type('array'):
        trainSet_X = numpy.array(trainSet_X)

        nDim = trainSet_X.shape[1]

        nSamples = len(trainSet_Y)
        if nSamples != trainSet_X.shape[0]:
            s = 'Number of training samples (%d) '\
            'is different from number of target values (%d)!\n' % (trainSet_X.shape[0], nSamples)
            raise ReadoutException(s)

        # 3. Find least squares solutions
        if self.addBias:
            trainSet_X = numpy.concatenate((trainSet_X, numpy.ones((trainSet_X.shape[0], 1))), 1)

        #print "executing the qr function"
        (Q, R)=scipy.linalg.qr(trainSet_X, overwrite_a=1)

        # now we solve the regression problems
        #print "executing the regression lstsquare algorithm"
        self.W = scipy.linalg.lstsq(R, numpy.dot(Q.T, trainSet_Y))[0]

        self.trained=True
        self.time = time.time() - t0


    def apply(self, X):
        if not self.trained:
            raise ReadoutException('readout is not trained!')

        X = numpy.array(X)

        if self.addBias:
            X = numpy.concatenate((X, numpy.ones((X.shape[0], 1))), 1)

        return utils.flatten(numpy.dot(X, self.W).tolist())


    def analyse(self, set_X, set_Y):
        ''' Analyse a trained linear classifier(s) on data
  Return values

    err   - classification error
    mse   - mean squared error (not very meaningful)
     cc   - correlation coefficient bewteen target and classifier output
    score - error measurement which takes into account the confusion matrix  
    CM    - the confusion matrix: CM(i,j) is the number of examples
            classified as class i while beeing actually class j.
    mi    - mutual information between 
            target and classifier output (calculated from CM).
    hx    - entropy of target values
    hy    - entropy of classifier output'''

        if not self.trained:
            raise ReadoutException('readout is not trained!')

        set_X = numpy.array(set_X)
        set_Y = copy.copy(set_Y)
        undef = numpy.where(numpy.isnan(set_Y))[0]

        # might be replaced by take()
        for ind in undef:
            set_X[ind,:] = []
            set_Y[ind] = []

        set_Y = numpy.asarray(set_Y)

        if numpy.prod(set_Y.shape) == 0:
            return (nan, nan, nan, nan, [], nan, nan, nan)

        # apply the classifier
        O = numpy.array(self.apply(set_X))

        scale=abs(numpy.diff(self.range))

        d = (O-set_Y)

#        mae=([abs(x) for x in d]).mean()     # mean classification error
        mae = abs(d).mean() # mean classification error
        mse = (d**2/scale**2).mean()              # mean squared error: not very meaningful      
        cc=corr_coeff(O, set_Y)                        # correlatio coefficient
        CM=confusion_matrix(O, set_Y, numpy.array([-1,1]))          # confusion matrx
        score=CM[0,1]/(1+CM[0,0]+CM[1,0]/(1+CM[1,1]))          # error score: takes into account false and positive negatives
        (mi,hx,hy,hxy)=mi_from_count(CM)                       # calculate mutual information and entropies between target and classifier output
        kappa =numpy.nan
        
        if cc is numpy.nan:
            print "Encountered nan for correlation coefficient"
            print "target output", O
            print "test output", set_Y
            

        return (mae,mse,cc,score,CM,mi,hx,hy, kappa)



class LinearNonNegRegression(LinearRegression):
    ''' Linear Non Negative Regression based on LMS'''
    def __init__(self, range=(-1,1), addBias=True, addNegBias=False, lsq_iter = 3):
        super(LinearNonNegRegression, self).__init__(range=range, addBias=addBias)
        self.time = -1
        self.addNegBias = addNegBias
        self.usefull_dims=None
        self.lsq_iter = lsq_iter

    def train(self, trainSet_X, trainSet_Y):
        '''train linear non negative regression'''
        t0 = time.time()
        trainSet_X = numpy.array(trainSet_X)

        sx = abs(trainSet_X).sum(0)
        sx_w=numpy.where(sx>0)[0]
        if len(sx_w) > 0:
            self.usefull_dims=sx_w
            trainSet_X=trainSet_X[:, self.usefull_dims]
        else:
            self.usefull_dims=None

        nDim = trainSet_X.shape[1]

        nSamples = len(trainSet_Y)
        if nSamples != trainSet_X.shape[0]:
            s = 'Number of training samples (%d) '\
            'is different from number of target values (%d)!\n' % (trainSet_X.shape[0], nSamples)
            raise ReadoutException(s)

        if self.addBias:
            trainSet_X = numpy.concatenate((trainSet_X, numpy.ones((trainSet_X.shape[0], 1))), 1)

        if self.addNegBias:
            trainSet_X = numpy.concatenate((trainSet_X, -1*numpy.ones((trainSet_X.shape[0], 1))), 1)            

        # now we solve the regression problems

        [self.W, resnorm, residual] = lsqnonneg(trainSet_X, trainSet_Y, itmax_factor = self.lsq_iter)

        self.trained=True
        self.time = time.time() - t0


    def apply(self, X):
        if not self.trained:
            raise ReadoutException('readout is not trained!')

        X = numpy.array(X)

        if self.usefull_dims is not None:
            X=X[:, self.usefull_dims]

        if self.addBias:
            X = numpy.concatenate((X, numpy.ones((X.shape[0], 1))), 1)
        if self.addNegBias:
            X = numpy.concatenate((X, -1*numpy.ones((X.shape[0], 1))), 1)

        return utils.flatten(numpy.dot(X, self.W).tolist())



    
if __name__=='__main__':
    trainSet_X = [[1,2,3],[1,3,3],[2,4,3],[1,1,3],[4,4,5]]
    trainSet_Y = [0,1,1,0,1]

    trainSet_X = [[1,2,3,2,2],[1,3,3,4,5],[1,4,3,4,5],[2,3,2,1,1],[1,2,3,1,1]]
    trainSet_Y = [0,1,2,0,2]

    trainSet_X = [[1],[2],[5],[6],[10]]
    trainSet_Y = [3, 5, 11, 13, 21]

    #testSet_X = [[1,2,4,3,1],[2,4,2,1,2],[1,3,3,3,1]]
    testSet_X = [[4],[3],[15]]
    testSet_Y = [9,7,3]

    trainSet_X0 = [[1,0,0],[2,0,0],[5,0,0],[6,0,0],[10,0,0]]
    testSet_X0 = [[4,0,0],[3,0,0],[15,0,0]]

    print 'Linear Regression:'
    lin_reg = LinearRegression()
    lin_reg.train(trainSet_X, trainSet_Y)
    testY = lin_reg.apply(testSet_X)

    print "training time: ", lin_reg.time
    print "W:", lin_reg.W
    (mae,mse,cc,score,CM,mi,hx,hy,kappa) = lin_reg.analyse(testSet_X, testSet_Y)
    print lin_reg.analyse(testSet_X, testSet_Y)

    print 'Linear Non Negative Regression:'
    lin_reg = LinearNonNegRegression()
    lin_reg.train(trainSet_X, trainSet_Y)
    testY = lin_reg.apply(testSet_X)
    
    print "training time: ", lin_reg.time
    print "W:", lin_reg.W
    (mae,mse,cc,score,CM,mi,hx,hy,kappa) = lin_reg.analyse(testSet_X, testSet_Y)
    print lin_reg.analyse(testSet_X, testSet_Y)


    print 'Linear Non Negative Regression (neg Bias):'
    lin_reg = LinearNonNegRegression(addNegBias=True)
    lin_reg.train(trainSet_X, trainSet_Y)
    testY = lin_reg.apply(testSet_X)

    print "training time: ", lin_reg.time
    print "W:", lin_reg.W
    (mae,mse,cc,score,CM,mi,hx,hy,kappa) = lin_reg.analyse(testSet_X, testSet_Y)
    print lin_reg.analyse(testSet_X, testSet_Y)


    print '\nlearning with zero value dimensions:'
    lin_reg0 = LinearNonNegRegression(addNegBias=True)
    lin_reg0.train(trainSet_X0, trainSet_Y)
    testY0 = lin_reg0.apply(testSet_X0)

    print "training time: ", lin_reg0.time
    print "W:", lin_reg0.W
    (mae,mse,cc,score,CM,mi,hx,hy,kappa) = lin_reg0.analyse(testSet_X0, testSet_Y)
    print lin_reg0.analyse(testSet_X0, testSet_Y)
