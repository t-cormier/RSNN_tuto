# Readout for SVM classification
#
# uses the PyML package
# http://pyml.sourceforge.net/
#
# Stefan Klampfl
# 2007/11/19

import numpy
from PyML import ker,svm,datafunc

from readout import *

class SVMReadout(Readout):
    def __init__(self, C=10, kernel=ker.Linear()):
        super(self.__class__, self).__init__()
        self.C = C
        self.kernel = kernel
        self.uniqueY = [0.0, 1.0]
        self.nClasses = len(self.uniqueY)

    # train classifier with samples X and labels Y
    def train(self, trainSet_X, trainSet_Y):
        trainSet_X = numpy.array(trainSet_X).astype(float)
        trainSet_Y = numpy.array(trainSet_Y).astype(float)
        labels = trainSet_Y.tolist()
        samples = trainSet_X.tolist()
        data = datafunc.VectorDataSet(samples, L=map(str,labels))
        self.svm = svm.SVM(self.kernel, C=self.C)
        self.svm.train(data)
        self.trained=True
        self.uniqueY = numpy.unique(labels)
        self.nClasses = len(self.uniqueY)

    # return class predictions for data X
    # X is an array or a sequence of data points
    # a list of class labels is returned
    def apply(self, X):
        if not self.trained:
            raise ReadoutException('readout is not trained!')

        X = numpy.array(X).astype(float).tolist()
        data = datafunc.VectorDataSet(X)
        return [float(self.svm.classify(data,i)[0]) for i in range(len(data))]

    # return decision values for data X
    # X is an array or a sequence of data points
    # a list of decision values is returned
    def fwd(self, X):
        X = numpy.array(X).astype(float).tolist()
        data = datafunc.VectorDataSet(X)
        return [self.svm.decisionFunc(data,i) for i in range(len(data))]

    # get weight and bias s.t. y = sgn(w*x+b) for linear SVMs
    def getwb(self):
        return (self.svm.model.w, self.svm.model.b)

    # calculate performance measures for test data X and labels Y
    def analyse(self, set_X, set_Y):
        if not self.trained:
            raise ReadoutException('readout is not trained!')

        set_X = numpy.array(set_X).astype(float)
        set_Y = numpy.array(set_Y).astype(float)
        undef = numpy.where(numpy.isnan(set_Y))[0]

        # might be replaced by take()
        for ind in undef:
            set_X[ind,:] = []
            set_Y[ind] = []

        if numpy.prod(set_Y.shape) == 0:
            return (nan, nan, nan, nan, [], nan, nan, nan)

        O = numpy.array(self.apply(set_X))

        mae = (O != set_Y.astype(numpy.double)).mean()        # mean classification error
        mse = ((O - set_Y.astype(numpy.double))**2).mean()    # mean squared error: not very meaningful here
        cc = corr_coeff(O, set_Y.astype(numpy.double))     # correlation coefficient
        CM = confusion_matrix(O, set_Y.astype(numpy.double), self.uniqueY) # confusion matrx
        score = CM[0,1]/(1+CM[0,0]+CM[1,0]/(1+CM[1,1]))
        (mi,hx,hy,hxy) = mi_from_count(CM)

        if self.nClasses == 2:
            kappa = ((1.0-mae)-0.5)/(1.0-0.5)
        else:
            kappa = numpy.nan

        return (mae, mse, cc, score, CM, mi, hx, hy, kappa)


if __name__=='__main__':
    #trainSet_X = [[1,2,3,2,2],[1,3,3,4,5],[1,4,3,4,5],[2,3,2,1,1]]
    trainSet_X = [[1,2,3],[1,3,3],[2,4,3],[1,1,3],[4,4,5]]
    trainSet_Y = [0,1,1,0,1]
    testSet_X = [[1,2,4],[2,4,2],[1,3,3]]
    testSet_Y = [0,1,1]

    print 'SVM Classification:'
    svm_readout = SVMReadout() 
    svm_readout.train(trainSet_X, trainSet_Y)
    testY = svm_readout.apply(testSet_X)
    fwdY = svm_readout.fwd(testSet_X)
    w,b = svm_readout.getwb()
    print "output:",testY, "target:",testSet_Y
    print "decisionFuncs:",fwdY
    print "w:",w, "b:",b
    print "w*x+b:", numpy.dot(w,numpy.array(testSet_X).transpose())+b
    print svm_readout.analyse(testSet_X, testSet_Y)
