import numpy

def rate2spikes(u, dt=1e-3, numChannels=1, Tmax=numpy.Inf):
    ''' calculates the instantaneous rate '''

    t = numpy.arange(len(u))*dt
    u = numpy.atleast_1d(u)

    spikes=[]

    for i in range(numChannels):
        #with prob u*dt*MaxFreq there is a spike in time bin [t,t+dt]
        st=t.compress((numpy.random.uniform(0, 1, len(u)) < u*dt).flat)

        #jitter spikes within interval [t,t+dt]
        if len(st) > 0:
            st = st + numpy.random.uniform(0, dt, len(st));

            # restrict to Tmax
            # now uses numpy.array, it is faster
            # st = [spike for spike in st if spike <= Tmax]
            st = st.compress((st <= Tmax).flat)

        spikes.append(st)

    return spikes





if __name__ == '__main__':
    rates = [10,20,20,10,10]
    spikes = rate2spikes(rates, dt=0.5, numChannels=2, Tmax=1.0)

    rates = numpy.array(rates)
    spikes = rate2spikes(rates, dt=0.5, numChannels=2, Tmax=1.0)
    
