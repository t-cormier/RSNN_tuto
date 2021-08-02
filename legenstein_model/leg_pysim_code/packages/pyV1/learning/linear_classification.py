import numpy
import utils
import scipy.linalg
import time
import copy
from readout import *
from lsqnonneg import lsqnonneg

import pdb

class LinearClassification(Readout):
    def __init__(self, nClasses = 2, addBias=True):
        super(LinearClassification, self).__init__()
        self.nClasses = nClasses
        self.addBias = addBias
        self.time = -1.0


    def train(self, trainSet_X, trainSet_Y):
        '''train linear classifier'''
        t0 = time.time()

        trainSet_X = numpy.array(trainSet_X)
        nDim = trainSet_X.shape[1]

        nSamples = len(trainSet_Y)
        if nSamples != trainSet_X.shape[0]:
            s='Number of training samples (%d) '\
            'is different from number of target values (%d)!\n' % (trainSet_X.shape[0], nSamples)
            raise ReadoutException(s)

        # 1. Convert the target values into range 1:nClasses => indexedY
        (self.uniqueY, indexedY) = unique_index(trainSet_Y)
        n = len(self.uniqueY);

        if n != self.nClasses:
            s='Number of actual class values (%d) is different from this.nClasses = %d!\n' % (len(self.uniqueY), self.nClasses)
            raise ReadoutException(s)

        # 2. make -1/+1 vector representation
        Y = numpy.zeros((nSamples,n))

        for i in range(n):
            l = numpy.where(numpy.asarray(indexedY) == i)[0]
            for j in l:
                Y[j,i] = 1

        Y = numpy.asarray(Y*2-1);

        if self.addBias:
            trainSet_X = numpy.concatenate((trainSet_X, numpy.ones((trainSet_X.shape[0],1))), 1)

        # 3. Find least squares solutions
        (Q, R)=scipy.linalg.qr(trainSet_X, overwrite_a=1)

        if self.nClasses > 2:    # now we solve the nClasses > 2 regression problems
            self.W = numpy.zeros((nDim+1, self.nClasses))
            for i in range(self.nClasses):
                self.W[:,i] = scipy.linalg.lstsq(R, numpy.dot(Q.T, Y[:,i]))[0]
        elif self.nClasses == 2:
            self.W = scipy.linalg.lstsq(R, numpy.dot(Q.T, Y[:,1]))[0]
        else:
            raise ReadoutException('nClasses == 1!?!?')

        self.trained=True
        self.time = time.time() - t0


    def apply(self, X):
        if not self.trained:
            raise ReadoutException('readout is not trained!')

        X = numpy.asarray(X)
        if self.addBias:
            X = numpy.concatenate((X, numpy.ones((X.shape[0], 1))), 1)

        if self.nClasses == 2:
            maxI = (numpy.dot(X, self.W)>=0).astype(numpy.int32)     # maxI \in {1,2}
        else:
            S = numpy.zeros((X.shape[0], self.nClasses))
            for i in range(self.nClasses):
                S[:,i] = numpy.dot(X, self.W[:,i])

            maxV=S.max(1)
            maxI=S.argmax(1)

        if maxI.ndim < 1: maxI = numpy.asarray([maxI])

        return utils.flatten(numpy.asarray(self.uniqueY).take(maxI).tolist())


    def analyse(self, set_X, set_Y):
        ''' LINEAR_CLASSIFICATION Analyse a trained linear classifier(s) on data

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
    hy    - entropy of classifier output
    '''
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
            return (nan, nan, nan, nan, [], nan, nan, nan, nan)

        O = self.apply(set_X)                 # apply the classifier

        mae = (O != set_Y.astype(numpy.double)).mean()        # mean classification error
        mse = ((O - set_Y.astype(numpy.double))**2).mean()    # mean squared error: not very meaningful here
        cc = corr_coeff(O, set_Y.astype(numpy.double))     # correlatio coefficient

        CM = confusion_matrix(O, set_Y.astype(numpy.double), self.uniqueY) # confusion matrx

        if self.nClasses == 2:
            kappa = ((1.0-mae)-0.5)/(1.0-0.5)
        else:
            kappa = numpy.nan

        # error score: takes into account false and positive negatives
        # only for 2 classes
        score = CM[0,1]/(1+CM[0,0]+CM[1,0]/(1+CM[1,1]))

        # calculate mutual information and entropies between target and classifier output
        (mi,hx,hy,hxy) = mi_from_count(CM)

        return (mae, mse, cc, score, CM, mi, hx, hy, kappa)



class LinearNonNegClassification(LinearClassification):
    def __init__(self, nClasses = 2, addBias=True, addNegBias=False, swapLabels=False):
        super(LinearNonNegClassification, self).__init__(nClasses=nClasses, addBias=addBias)
        self.addNegBias = addNegBias
        self.usefull_dims = None
        self.swapLabels = swapLabels

    def apply(self, X):
        if not self.trained:
            raise ReadoutException('readout is not trained!')

        X = numpy.array(X)

        if self.usefull_dims is not None:
            X = X[:, self.usefull_dims]

        if self.addBias:
            X = numpy.concatenate((X, numpy.ones((X.shape[0], 1))), 1)
        if self.addNegBias:
            X = numpy.concatenate((X, -1*numpy.ones((X.shape[0], 1))), 1)

        if self.nClasses == 2:
            if self.addNegBias:
                maxI = (numpy.dot(X, self.W)>=0).astype(numpy.int32)     # maxI \in {1,2}
            else:
                maxI = (numpy.dot(X, self.W)>=0.5).astype(numpy.int32)     # maxI \in {1,2}
        else:
            S = numpy.zeros((X.shape[0], self.nClasses))
            for i in range(self.nClasses):
                S[:,i] = numpy.dot(self.W[i,:], X.T)

            maxV=S.max(1)
            maxI=S.argmax(1)

        if maxI.ndim < 1:
            maxI = numpy.asarray([maxI])

        Y=numpy.array(utils.flatten(numpy.asarray(self.uniqueY).take(maxI).tolist()))

        if self.swapLabels:
            Y = (numpy.array(Y*2-1)*(-1) + 1)/2;

        return Y.tolist()


    def train(self, trainSet_X, trainSet_Y):
        '''train linear classifier'''
        t0 = time.time()
        # convert to matrix
        trainSet_X = numpy.array(trainSet_X)

        sx = abs(trainSet_X).sum(0)
        sx_w=numpy.where(sx>0)[0]

        if len(sx_w) > 0:
            self.usefull_dims = sx_w
            trainSet_X = trainSet_X[:, self.usefull_dims]
        else:
            self.usefull_dims=None

        nDim = trainSet_X.shape[1]
        trainSet_Y = numpy.array(trainSet_Y)

        nSamples = len(trainSet_Y)
        if nSamples != trainSet_X.shape[0]:
            s='Number of training samples (%d) '\
            'is different from number of target values (%d)!\n' % (trainSet_X.shape[0], nSamples)
            raise ReadoutException(s)

        # 1. Convert the target values into range 1:nClasses => indexedY
        (self.uniqueY, indexedY) = unique_index(trainSet_Y)
        n = len(self.uniqueY);

        if n != self.nClasses:
            s='Number of actual class values (%d) is different from this.nClasses = %d!\n' % (len(self.uniqueY), self.nClasses)
            raise ReadoutException(s)

        # 2. make -1/+1 vector representation
        Y = numpy.zeros((nSamples,n))

        for i in range(n):
            l = numpy.where(numpy.asarray(indexedY) == i)[0].tolist()
            for j in l:
                Y[j,i] = 1

        if self.swapLabels:
            Y = (numpy.array(Y*2-1)*(-1) + 1)/2;

        if self.addNegBias:
            Y = numpy.array(Y*2-1);

        if self.addBias:
            trainSet_X = numpy.concatenate((trainSet_X, numpy.ones((trainSet_X.shape[0],1))), 1)

        if self.addNegBias:
            trainSet_X = numpy.concatenate((trainSet_X, -1.0*numpy.ones((trainSet_X.shape[0], 1))), 1)

        nDim = trainSet_X.shape[1]

        # 3. Find least squares solutions
        if self.nClasses > 2:
            # now we solve the nClasses > 2 regression problems
            self.W = numpy.zeros((self.nClasses, nDim))

            for i in range(self.nClasses):
                self.W[i,:] = lsqnonneg(trainSet_X, Y[:,i])[0]

        elif self.nClasses == 2:
            self.W = lsqnonneg(trainSet_X, Y[:,1])[0]
        else:
            raise ReadoutException('nClasses == 1!?!?')

        self.trained=True
        self.time = time.time() - t0

if __name__=='__main__':

    trainSet_X = [[1,2,3,2,2],[1,3,3,4,5],[1,4,3,4,5],[2,3,2,1,1]]
    trainSet_X = [[1,2,3],[1,3,3],[2,4,3],[1,1,3],[4,4,5]]
    trainSet2_Y = [0,1,1,0,1]
    trainSet3_Y = [0,1,2,0,2]
    testSet_X = [[1,2,4],[2,4,2],[1,3,3]]
    testSet2_Y = [0,1,1]
    testSet3_Y = [0,1,2]
    trainSet_X0 = [[1,2,0,3],[1,3,0,3],[2,4,0,3],[1,1,0,3],[4,4,0,5]]
    testSet_X0 = [[1,2,0,4],[2,4,0,2],[1,3,0,3]]

    print 'Linear Classification:'
    print '\n2 Classes:'
    lin_readout2 = LinearClassification(2)
    lin_readout2.train(trainSet_X, trainSet2_Y)
    testY2 = lin_readout2.apply(testSet_X)
    print lin_readout2.analyse(testSet_X, testSet2_Y)
    print "training time 2 classes: ", lin_readout2.time
    print "W:", lin_readout2.W

    print '\n3 Classes:'
    lin_readout3 = LinearClassification(3)
    lin_readout3.train(trainSet_X, trainSet3_Y)
    testY3 = lin_readout3.apply(testSet_X)
    print lin_readout3.analyse(testSet_X, testSet3_Y)
    print "training time 3 classes: ", lin_readout3.time
    print "W:", lin_readout3.W

    # make training data tr
    print '\nrandom 2 Classes:'
    num_samples = 50
    num_features = 11
    tr_X=numpy.random.uniform(0, 1, (num_samples, num_features))-0.5
    tr_Y=(tr_X.sum(1) > 0).astype(numpy.int32)

    # do the training
    LC = LinearClassification()
    LC.train(tr_X, tr_Y);
    print "training time random samples: ", LC.time
    print "W:", LC.W

    # make test data te
    te_X=numpy.random.uniform(0,1, (num_samples, num_features))-0.5
    te_Y=(tr_X.sum(1) > 0).astype(numpy.int32)

    # apply the trained classifier and measure error
    (mae,mse,cc,score,CM,mi,hx,hy,kappa) = LC.analyse(te_X, te_Y)

    print '\n\nLinear Non Negative Classification:'
    print '\nrandom 2 Classes:'

    lin_nnreadout2 = LinearNonNegClassification(2, addNegBias=False)
    lin_nnreadout2.train(trainSet_X, trainSet2_Y)
    print lin_nnreadout2.analyse(testSet_X, testSet2_Y)
    print "training time 2 classes: ", lin_nnreadout2.time
    print "W:", lin_nnreadout2.W

    lin_nnreadout2neg = LinearNonNegClassification(2, addNegBias=True)
    lin_nnreadout2neg.train(trainSet_X, trainSet2_Y)
    print lin_nnreadout2neg.analyse(testSet_X, testSet2_Y)
    print "training time 2 classes (neg): ", lin_nnreadout2neg.time
    print "W:", lin_nnreadout2neg.W

    lin_nnreadout3 = LinearNonNegClassification(3, addNegBias=True)
    lin_nnreadout3.train(trainSet_X, trainSet3_Y)
    testY3 = lin_nnreadout3.apply(testSet_X)
    print lin_nnreadout3.analyse(testSet_X, testSet3_Y)
    print "training time 3 classes: ", lin_nnreadout3.time
    print "W:", lin_nnreadout3.W

    print '\nlearning with zero value dimensions:'
    lin_nnreadout4 = LinearNonNegClassification(3, addNegBias=True)
    lin_nnreadout4.train(trainSet_X0, trainSet3_Y)
    testY3 = lin_nnreadout4.apply(testSet_X0)
    print lin_nnreadout4.analyse(testSet_X0, testSet3_Y)
    print "training time 3 classes: ", lin_nnreadout3.time
    print "W:", lin_nnreadout3.W
