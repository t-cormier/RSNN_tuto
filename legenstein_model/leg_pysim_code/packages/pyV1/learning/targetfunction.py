from scipy.interpolate.interpolate import interp1d

import numpy
import pyV1.utils

from pyV1.inputs.jitteredtemplate import *
from pyV1.inputs.constantrate import *
from pyV1.inputs.randomrate import *
from pyV1.inputs.randombndrate import *
from pyV1.inputs.randommodrate import *

from pyV1.inputs.spikes2rate import spikes2rate

import scipy.interpolate

#import pdb as pdbm

class TargetFunction(object):
    def __init__(self, pos_gen=None):
        self.pos_gen=pos_gen

    def values(self, stim, at_t):
        pass

    def getStim(self, stim):
        if isinstance(stim, CombinedStimulus):
            if not self.pos_gen is None:
                return stim.stim_list[self.pos_gen]

        return stim


class SegmentClassification(TargetFunction):
    '''segment classification'''

    def __init__(self, posSeg=0, pos_gen=None):
        '''posSeg ... Segment position to classify
pos_gen ... which generator to classify (only if CombinedInput)
'''
        super(SegmentClassification, self).__init__(pos_gen)
        self.posSeg=posSeg            # number of segment to classifiy as positive


    def values(self, stim, at_t):
        s = super(SegmentClassification, self).getStim(stim)

        try:
            nSegments=len(s.actualTemplate)
        except:
            raise Exception('Given stimulus is not a template stimulus!')

        if self.posSeg >= nSegments:
            raise Exception('segment out of range!')

        if -self.posSeg > nSegments:
            raise Exception('segment out of range!')

#        if self.pos_gen is None:
        v=(s.actualTemplate[self.posSeg])

        return numpy.tile(v, len(at_t))



class SumOfRates(TargetFunction):
    '''sum of rates (normalized) in the time window [t-delay-W,t-delay]'''
    
    def __init__(self, delay=0e-3, fmax=100, input_binwidth=15e-3, nValues=numpy.inf, channels=None, pos_gen=None, norm=False):
        super(SumOfRates, self).__init__(pos_gen)
        self.delay=delay            # the delay of the time window [t-delay-W,t-delay]
        self.fmax=fmax              # the maximal frequency assumed to occur
        self.nValues=nValues        # number of possible values of the target function (''Inf'' for real valued)
        self.input_binwidth=input_binwidth
        self.norm=norm

        if channels is None:
            channels = []

        self.channels=channels      # over which channels do calculate the sum of rates


    def values(self, input, at_t):
        stim = super(SumOfRates, self).getStim(input)

        # merge all spikes to a single spike train spikes
        if len(self.channels)>0:
            spikes=utils.flatten([[stim.channel[j].data for j in self.channels], input.Tsim])
            d=len(self.channels)
        else:
            spikes=utils.flatten([[c.data for c in stim.channel], input.Tsim])
            d=len(stim.channel)

#        spikes = numpy.hstack((spikes, input.Tsim))
        mfr = len(spikes)/input.Tsim/d

        (rates, t)=spikes2rate(numpy.asarray(spikes), self.input_binwidth, dt=1e-3)
        rates /= d
        if self.norm:
            rates /= self.fmax

        ylin = scipy.interpolate.interp1d(t, rates)
        tmax=t.max()
        tmin=t.min()
        att = numpy.array(at_t)-self.delay
        att[att>tmax]=tmax
        att[att<tmin]=tmin

        y = numpy.asarray(ylin(att))

        return y



class SpikeCorrelation(TargetFunction):
    '''correlation of spikes within the interval [t-delay-W,t-delay]'''

    def __init__(self, delay=0e-3, W=50e-3, delta=5e-3, channels=None, pos_gen=None):
        super(SpikeCorrelation, self).__init__(pos_gen)
        self.delay=delay            # the delay of the time window [t-delay-W,t-delay]
        self.W=W                    # the width of the time window [t-delay-W,t-delay]
        self.delta=delta            # precision of coincidence detection

        if channels is None:
            channels = []

        self.channels=channels      # over which channels do calculate the sum of rates
        
    
    def values(self, input, at_t):
        stim = super(SpikeCorrelation, self).getStim(input)

        if len(self.channels)>0:
            spikes = utils.flatten([stim.channel[j].data for j in self.channels])
        else:
            spikes = utils.flatten([c.data for c in stim.channel])

        spikes.sort()
        n=len(self.channels)

        s=numpy.ones(n)
        ii=0
        tco=numpy.nan*numpy.ones(len(spikes))

        for t in spikes:
            c=numpy.zeros(n)

            for i in range(n):
                st = stim.channel(this.channels(i)).data
                strain = st.compress((st >= (t-self.delta)).flat)

                if len(strain) > 0:
                    if strain[0] <= t:
                        c[i] = c[i]+1

#                stim.channel[self.channels[i]].data=st

            if numpy.all(c>0):
                tco[ii]=t
                ii=ii+1

        tco=tco.compress((tco != numpy.nan).flat)

        y=numpy.zeros(len(at_t))

        for i in range(len(at_t)):
            a=(at_t[i]-self.W-self.delay) < tco
            b=tco <= (at_t[i]-self.delay)
            y[i]=((numpy.logical_and(a, b)).astype(numpy.int32)).sum()

        return y



class CombinedTarget(TargetFunction):
    '''combine arbitary target functions'''
    
    def __init__(self, targets=[], expr='f1'):
        self.targets=targets        # list of target function to be combined
        self.expr=expr              # a string like ''f1*f2+sin(f3)'' which defines how to combine the target functions f1,f2,..


    def values(self, input, at_t):
#        pdbm.set_trace()

        if len(self.targets) > 0:
            i=1
            for target in self.targets:
                f=numpy.array(target.values(input, at_t))
                exec('f%d = f' % i)
                i+=1

            y=eval(self.expr)
        else:
            y=numpy.nan*numpy.ones(len(at_t))
            
        return numpy.atleast_1d(y)
    

if __name__=='__main__':
    Tstim=1.0
    bin = 0.2
    times = numpy.arange(0, Tstim + bin/2.0, bin)
    
    jtemp=JitteredTemplate(Tstim=Tstim, nChannels=2, nTemplates=[2, 2], jitter=4e-3, freq=[5,10])
    jtemp_multi=JitteredTemplate(Tstim=Tstim, nChannels=3, nTemplates=[3, 3], jitter=4e-3, freq=[10,5,10])
    crate=ConstantRate(Tstim=Tstim)
    
    stim_seg=jtemp.generate()
    stim_seg_multi=jtemp_multi.generate()
    stim_rates=crate.generate()
    
    class_seg0=SegmentClassification(posSeg=0)    
    v_seg0=class_seg0.values(stim_seg, times)
    print "SegmentClassification (posSeg=0) values:", v_seg0, "\n"

    class_seg1=SegmentClassification(posSeg=1)
    v_seg1=class_seg1.values(stim_seg, times)
    print "SegmentClassification (posSeg=1) values:", v_seg1, "\n"

    class_seg0_multi=SegmentClassification(posSeg=0)
    v_seg0_multi=class_seg0_multi.values(stim_seg_multi, times)
    print "SegmentClassification (posSeg=0, multi) values:", v_seg0_multi, "\n"

    class_seg1_multi=SegmentClassification(posSeg=1)
    v_seg1_multi=class_seg1_multi.values(stim_seg_multi, times)
    print "SegmentClassification (posSeg=1, multi) values:", v_seg1_multi, "\n"

    try:
        class_seg2_multi=SegmentClassification(posSeg=2)
        v_seg2_multi=class_seg2_multi.values(stim_seg_multi, times)
        print "SegmentClassification (posSeg=2, multi) values:", v_seg2_multi, "\n"
    except:
        print 'Exception: ok'

    try:
        class_seg2_multi=SegmentClassification(posSeg=-3)
        v_seg2_multi=class_seg2_multi.values(stim_seg_multi, times)
        print "SegmentClassification (posSeg=2, multi) values:", v_seg2_multi, "\n"
    except:
        print 'Exception: ok'

    class_seg0_multi=SegmentClassification(posSeg=0)
    v_seg0_multi=class_seg0_multi.values(stim_seg_multi, times)
    print "SegmentClassification (posSeg=0, multi) values:", v_seg0_multi, "\n"

    spike_corr=SpikeCorrelation()
    v_corr=spike_corr.values(stim_rates, times)
    print "SpikeCorrelation values:", v_corr, "\n"

    sum_rates=SumOfRates()
    v_sumrates=sum_rates.values(stim_rates, times)
    print "SumOfRates values:", v_sumrates, "\n"

    sum_rates_norm=SumOfRates(norm=True)
    v_sumrates_norm=sum_rates_norm.values(stim_rates, times)
    print "SumOfRates (normalized) values:", v_sumrates_norm, "\n"

#    combined=CombinedTarget(targets=[SumOfRates(channels=[0]), SumOfRates(channels=[1]),
#                                    SumOfRates(channels=[2]), SumOfRates(channels=[3])],
#                                    expr='f1+f2+f3+f4')
#    v_comb=combined.values(stim_rates, times)
#    print "Combined values:", v_comb, "\n"
    
    
    jtemp1=JitteredTemplate(Tstim=Tstim, nChannels=2, nTemplates=[2, 2], jitter=4e-3, freq=[5,10])
    jtemp2=JitteredTemplate(Tstim=Tstim, nChannels=2, nTemplates=[2, 2], jitter=4e-3, freq=[5,10])
    comb_jtemp_gen=CombinedInputGenerator([jtemp1, jtemp2])

    comb_jtemp_stim=comb_jtemp_gen.generate()

    class_seg0_gen0=SegmentClassification(posSeg=-2, pos_gen=0)
    v_seg0_gen0=class_seg0_gen0.values(comb_jtemp_stim, times)
    print "SegmentClassification (posSeg=0, pos_gen=0) values:", v_seg0_gen0
    print 'actualTemplate:', comb_jtemp_stim.stim_list[0].actualTemplate, '\n'

    class_seg0_gen1=SegmentClassification(posSeg=-2, pos_gen=1)
    v_seg0_gen1=class_seg0_gen1.values(comb_jtemp_stim, times)
    print "SegmentClassification (posSeg=0, pos_gen=1) values:", v_seg0_gen1
    print 'actualTemplate:', comb_jtemp_stim.stim_list[1].actualTemplate, '\n'

    xor_target = CombinedTarget(targets=[SegmentClassification(posSeg=(0), pos_gen=0), \
                                         SegmentClassification(posSeg=(0), pos_gen=1)], expr='(f1+f2) % 2')
    v_xor_target=xor_target.values(comb_jtemp_stim, times)
    print "xor_target:", v_xor_target, "\n"


    fmax1 = 25; fmax2 = 25; fmin1 = 15; fmin2 = 15;
    rate_gen1=RandomBndRate(Tstim=Tstim, binwidth=bin, nChannels=2, fmin=fmin1, fmax=fmax1)
    rate_gen2=RandomBndRate(Tstim=Tstim, binwidth=bin, nChannels=2, fmin=fmin2, fmax=fmax2)
    comb_rate_gen=CombinedInputGenerator([rate_gen1, rate_gen2])


    comb_rate_stim=comb_rate_gen.generate()

    comb_rate_target=CombinedTarget(targets=[SumOfRates(pos_gen=0,delay=0.004), SumOfRates(pos_gen=1, delay=0.004)], expr='f1/f2')
    v_comb_target=comb_rate_target.values(comb_rate_stim, times)
    print "Combined values (f1/f2):", v_comb_target
    sum_rates_delay=SumOfRates(delay=0.004)
    v_comb_rate=sum_rates_delay.values(comb_rate_stim, times)
    print "Combined values sum_rates all:", v_comb_rate
    sum_rates_gen0=SumOfRates(pos_gen=0, delay=0.004)
    v_comb_rate_gen0=sum_rates_gen0.values(comb_rate_stim, times)
    print "Combined values sum_rates 0:", v_comb_rate_gen0
    sum_rates_gen1=SumOfRates(pos_gen=1, delay=0.004)
    v_comb_rate_gen1=sum_rates_gen1.values(comb_rate_stim, times)
    print "Combined values sum_rates 1:", v_comb_rate_gen1
    print "Combined values (f1/f2) test:", v_comb_rate_gen0/v_comb_rate_gen1, "\n"

    comb_rate_target=CombinedTarget(targets=[SumOfRates(pos_gen=0), SumOfRates(pos_gen=1)], expr='f1/f2')
    v_comb_target=comb_rate_target.values(comb_rate_stim, times)
    print "Combined values (f1/f2):", v_comb_target
    sum_rates=SumOfRates()
    v_comb_rate=sum_rates.values(comb_rate_stim, times)
    print "Combined values sum_rates all:", v_comb_rate
    sum_rates_gen0=SumOfRates(pos_gen=0)
    v_comb_rate_gen0=sum_rates_gen0.values(comb_rate_stim, times)
    print "Combined values sum_rates 0:", v_comb_rate_gen0
    sum_rates_gen1=SumOfRates(pos_gen=1)
    v_comb_rate_gen1=sum_rates_gen1.values(comb_rate_stim, times)
    print "Combined values sum_rates 1:", v_comb_rate_gen1
    print "Combined values (f1/f2) test:", v_comb_rate_gen0/v_comb_rate_gen1, "\n"
