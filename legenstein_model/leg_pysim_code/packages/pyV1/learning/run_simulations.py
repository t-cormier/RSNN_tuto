# functions for running network simulations and collecting stimuli, responses and states
#
# Stefan Klampfl
# 2007/11/21
#

import sys

# runs several simulations and returns stimuli and responses on a single machine
# net: the network instance
# indist: the input distribution
# nS: number of simulations
# Tsim: duration of simulation (-1: take length of stimulus)
def runSimulations(net, indist, nS, Tsim=-1):
    print 'running', nS, 'simulations with input distribution', indist.__class__.__name__

    stimulus=[]
    response=[] 
    for i in range(nS):
        sys.stdout.write('.')
        stim = indist.generate(i)
        stimulus.append(stim)

        # check simulation time
        if Tsim == -1:
            Tmax=stim.Tsim
        else:
            Tmax=Tsim

        if Tmax < 0:
            raise Exception('Neither stimulus determines Tsim nor is Tsim explicitly given.')

        resp = net.simulate(stimulus=stim, Tsim=Tmax)
        if type(resp)==tuple:
            resp = resp[0]
        response.append(resp)
        if (i+1)%50==0 or i==nS-1:
            sys.stdout.write(" %d/%d\n" % (i+1,nS))
        sys.stdout.flush()
    return (stimulus, response)

# runs several simulations on different machines and returns stimuli and states
# tc: an ipython1 task controller instance
# netgenstr: string with python code that evaluates to a network instance
# params: additional parameter instance that can be used in netgenstr
# indist: the input distribution
# nS: number of simulations
# tstr: a string representing the sample times for the states in slice notation
# Tsim: duration of simulation (-1: take length of stimulus)
def runDistributedSimulations(tc, netgenstr, params, indist, nS, tstr, Tsim=-1):
    import ipython1.kernel.api as kernel
    print 'running', nS, 'simulations with input distribution', indist.__class__.__name__

    stimuli=[]
    states=[]
    taskIds = dict()
    sys.stdout.write("Generating tasks:   0%")
    for i in range(nS):
        cmd="""
from area import Area
from area_spatial import AreaSpatial
from learning import response2states as r2s

stimulus = indist.generate(i)
if Tsim==-1:
    Tmax = stimulus.Tsim
else:
    Tmax = Tsim
net = eval(netgenstr)
net.generate()
response,aresp = net.simulate(stimulus=stimulus, Tsim=Tmax)
states = r2s.response2states([response], sampling=tstr)[0]
"""
        taskIds[i] = tc.run(kernel.Task(cmd, resultNames=['stimulus','states'], clearBefore=True,
            setupNS={'i':i,'netgenstr':netgenstr,'params':params,'indist':indist,'tstr':tstr,'Tsim':Tsim}))
        sys.stdout.write("\b\b\b\b%3d%%" % ((i+1)*100/nS))
        sys.stdout.flush()
    sys.stdout.write("\b\b\b\b100%\n")
    #tc.barrier(taskIds.values())
    for i in range(nS):
        sys.stdout.write('.')
        result = tc.getTaskResult(taskIds[i], block=True)
        stimuli.append(result.ns.stimulus)
        states.append(result.ns.states)
        if (i+1)%50==0 or i==nS-1:
            sys.stdout.write(" %d/%d\n" % (i+1,nS))
        sys.stdout.flush()
    return (stimuli, states)

