from scipy.interpolate.interpolate import interp1d
import numpy
import pyV1.utils

def unique(spikes):
    s=list(spikes[:])
    s.sort()
    return s

#@utils.timedTest
def spikes2rate_new(spikes, binwidth=1e-2, dt=1e-3):
    s = numpy.atleast_1d(spikes)
    s.sort()

    t1 = numpy.r_[0, s]

    t2 = s + binwidth
    t2 = numpy.r_[0, t2]

    r = numpy.arange(len(s)+1)
    r1 = interp1d(t1, r, 'linear')
    r2 = interp1d(t2, r, 'linear')

    Tmax = s[-1]
    t=numpy.arange(0, Tmax+dt/2.0, dt)
    rate = (r1(t) - r2(t))/binwidth

    return (rate, t)


#@utils.timedTest
def spikes2rate(spikes, binwidth=1e-2, dt=1e-3):

    s = unique(spikes[:])

    t1 = s[:]
    t1.insert(0, 0)
#    print 't1:', t1

    t2 = [(spike+binwidth) for spike in s]
    t2.insert(0, 0)
#    print 't2:', t2

    r = range(len(s)+1)
    r1 = interp1d(t1, r, 'linear')
    r2 = interp1d(t2, r, 'linear')

    Tmax = s[-1]
    t=numpy.arange(0,Tmax,dt)
    rate = (r1(t) - r2(t))/binwidth

    return (rate, t)


def channels2rate(channels, binwidth=1e-2, dt=1e-3):
    return spikes2rate(utils.flatten([c.data for c in channels if c.spiking]), binwidth, dt)    



if __name__ == '__main__':
    Tsim=0.45
    freq=40
    nspikes=numpy.random.uniform(0, Tsim, freq*Tsim)
    spikes=nspikes.tolist()+ [Tsim]

#    spikes = [0.1, 0.15, 0.22, 0.34, 0.28, 0.41]
#    nspikes = numpy.array(spikes)

    binwidth=15e-3
    dt=1e-3
    s = numpy.atleast_1d(spikes)
    s.sort()

    t1 = numpy.r_[0, s]

    t2 = s + binwidth
    t2 = numpy.r_[0, t2]

    r = numpy.arange(len(s)+1)
    r1 = interp1d(t1, r, 'linear')
    r2 = interp1d(t2, r, 'linear')

    Tmax = s[-1]
    t=numpy.arange(0, Tmax+dt/2.0, dt)
    rate = (r1(t) - r2(t))/binwidth

#    print "Old version:"
#    (rate_old, t_old) = spikes2rate_old(spikes, binwidth=0.015, dt=1e-3)
#    print "rate:", rate

#    print "\nNumpy version:"
#    (rate1,t1) = spikes2rate(spikes, binwidth=0.015, dt=1e-3)
#    print "rate:", rate
