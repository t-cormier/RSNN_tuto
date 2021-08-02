import os, sys
sys.path.append('../packages')
from pylab import *
from tables import *
from numpy import *
from math import *
from pypcsimplus import *
from frame import FrameAxes
import numpy

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('text', usetex=True)
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})

# Load results

if len(sys.argv) > 1:
    h5filename = sys.argv[1]
else:
    h5filename = last_file(".*\.h5")
    
print " loading h5 filename : ", h5filename

h5file = openFile(h5filename, mode = "r", title = "Biofeedback DASTDP Experiment results")

p = constructParametersFromH5File(h5file)
r = constructRecordingsFromH5File(h5file)

print p


A_r_plus = p.stdpApos * p.DAStdpRate * (p.rewardScale / (p.rateTau * exp(1.0))) / (p.DTsim * 0.01 * 2 * p.Wexc)

print "VALUE OF THE A_R_plus_theory :" , A_r_plus
print "VALUE OF THE A_R_minus_theory :", A_r_plus * p.rateTau / p.negTau
print "VALUE OF THE A_plus_theory :", 0.01 * 2 * p.Wexc


other_circ_not_ou_weights = r.other_circ_not_ou_weights.copy()
other_circ_ou_weights = r.other_circ_ou_weights.copy()

other_circ_ou_weights *= 1.0/p.WHighOUScale
other_circ_not_ou_weights *= 1.0/p.WLowOUScale


other_weights = vstack((other_circ_not_ou_weights, other_circ_ou_weights))    

reinforced_circ_weights = vstack((r.reinforced_ou_weights, r.reinforced_other_weights))

reinforced_circ_weights *= 1.0/p.WLowOUScale

p.Wmax = 2 * p.Wexc

exc_ou_spikes = []
for i in r.exc_ou_nrn_idxs:
    exc_ou_spikes.append(r.spikes[i])
    
exc_other_spikes = []
for i in r.exc_other_nrn_idxs:
    exc_other_spikes.append(r.spikes[i])


reinforced_circ_avg_weights = average(reinforced_circ_weights, 0)
reinforced_circ_std_weights = std(reinforced_circ_weights, 0)

reinforced_ou_avg_weights = average(r.reinforced_ou_weights, 0)
reinforced_ou_std_weights = std(r.reinforced_ou_weights, 0)

reinforced_other_avg_weights = average(r.reinforced_other_weights, 0)
reinforced_other_std_weights = std(r.reinforced_other_weights, 0)

other_avg_weights = average(other_weights, 0)
other_std_weights = std(other_weights, 0)


other_circ_avg_not_ou_weights = average(other_circ_not_ou_weights, 0)
other_circ_std_not_ou_weights = std(other_circ_not_ou_weights, 0)

other_circ_avg_ou_weights = average(other_circ_ou_weights, 0)
other_circ_std_ou_weights = std(other_circ_ou_weights, 0)


f = figure(1,figsize=(8,8), facecolor = 'w')

f.subplots_adjust(top= 0.95, left = 0.1, bottom = 0.07, right = 0.95, hspace = 0.76, wspace = 0.54)
clf()

# plot liquid r.spikes
nRespNeurons = 99
leftX = 20; rightX = 23 
gap_duration = 0.5

numpy.random.seed(34223515)

chosenNeuronsArray = numpy.random.permutation(p.nNeurons)[:nRespNeurons]
chosenNeuronsArraySplit1 = chosenNeuronsArray[:int(nRespNeurons / 2)]
chosenNeuronsArraySplit2 = chosenNeuronsArray[int(nRespNeurons / 2):]
chosenNeuronsArray = hstack((chosenNeuronsArraySplit1,p.reinforced_nrn_idx,chosenNeuronsArraySplit2))

chosenSpikes = [ r.spikes[i] for i in chosenNeuronsArray ]

reinfNeuronRasterIdx = int(nRespNeurons / 2)

raster_x, raster_y = create_raster(chosenSpikes, leftX, rightX, shift = True)        
clipped_spikes_reinforc = clip_window(chosenSpikes[reinfNeuronRasterIdx], leftX, rightX, shift = True)

leftX_2, rightX_2 = p.Tsim - 3, p.Tsim
raster_x_2, raster_y_2 = create_raster(chosenSpikes,leftX_2, rightX_2, shift = True)
raster_x_2 += (rightX - leftX) + gap_duration
clipped_spikes_reinforc_2 = clip_window(chosenSpikes[reinfNeuronRasterIdx], leftX_2, rightX_2, shift = True) + (rightX - leftX) + gap_duration 

total_raster_x = hstack((raster_x, raster_x_2))
total_raster_y = hstack((raster_y, raster_y_2))

total_clip_spikes_reinforc = hstack((clipped_spikes_reinforc, clipped_spikes_reinforc_2))
 
ax = subplot(3, 1, 1, projection = 'frameaxes')
plot(total_raster_x, total_raster_y, '.', color = '0.0', markersize = 0.3)
plot(total_clip_spikes_reinforc, [ reinfNeuronRasterIdx for i in range(len(total_clip_spikes_reinforc)) ], '+', color = '0.0', markersize = 10, mec =  'b', mfc = 'b',  markeredgewidth = 2)


TickStep = 1.0    

xticks( hstack((arange(0, rightX - leftX + 0.01, TickStep), 
                arange(rightX -leftX + gap_duration, rightX - leftX + gap_duration + (rightX_2 - leftX_2) + 0.1, TickStep))), 
         [ "%d" % (x,) for x in arange(0, rightX - leftX + 0.01, TickStep) ] + [ "%d" % (x,) for x in arange(0, rightX_2 - leftX_2 + 0.1, TickStep) ] )
yticks([])



ylabel('100 neurons')
text(-0.09, 1.07, 'A', fontsize = 'x-large', transform = ax.transAxes )

axvline(rightX - leftX + gap_duration, color = 'k')


text(0.17, -0.33, 'time [sec]', transform = ax.transAxes )
text(0.7, -0.33, 'time [sec]', transform = ax.transAxes )


axhline(-1, 3.1/(rightX - leftX + rightX_2 - leftX_2  + gap_duration + 0.1), (2.97 + gap_duration)/(rightX - leftX + rightX_2 - leftX_2  + gap_duration + 0.1), linestyle = '-', color = (1,1,1), linewidth = 3)
xlim(0, rightX - leftX + rightX_2 - leftX_2  + gap_duration + 0.1)

    

ax = subplot(3, 2, 3, projection = 'frameaxes')

reinforced_nrn_rate = calc_rate_2(r.spikes[p.reinforced_nrn_idx], int(p.Tsim/(p.samplingTime * p.DTsim / 4)), 40, p.Tsim)

nrn_rate = []
for s in r.spikes[0:20]:
    nrn_rate.append(calc_rate_2(s, int(p.Tsim/(p.samplingTime * p.DTsim / 4)), 40, p.Tsim))
avg_nrn_rate = []
for i in range(len(nrn_rate[0])):
    avg_nrn_rate.append(sum([ nrn_rate[k][i] for k in range(len(nrn_rate)) ]) / len(nrn_rate))    
plot(arange(0, (len(reinforced_nrn_rate) - .5) * p.DTsim * p.samplingTime / 4, p.DTsim * p.samplingTime / 4), reinforced_nrn_rate, 'b-', linewidth = 1.2)
plot(arange(0, (len(avg_nrn_rate) - .5) * p.DTsim * p.samplingTime / 4, p.DTsim * p.samplingTime / 4), avg_nrn_rate, 'k--', linewidth = 1.2)
theGap = 10 * p.DTsim * p.samplingTime + 10 * p.DTsim * p.samplingTime / 4
print "theGap is", theGap
xlim(0, p.Tsim - theGap + p.Tsim/1000)
tickInterval = 300

xticks( array([ i * tickInterval - theGap for i in range(1, int(p.Tsim/tickInterval) + 1) ]), [ '%d' % (i * tickInterval / 60)  for i in range(1, int(p.Tsim/tickInterval) + 1) ])
xlabel("time [min]")
ylim(4,13)
yticks(arange(4,14,2), [ '%d' % (x,) for x in arange(4,14,2) ])
ylabel("rate [Hz]")
text(-0.24, 1.15, 'B', fontsize = 'x-large', transform = ax.transAxes )




ax = subplot(3, 2, 4, projection = 'frameaxes')
plot(arange(0, (len(other_avg_weights) -0.5) * p.DTsim*p.samplingTime, p.DTsim*p.samplingTime), other_avg_weights, 'k--', linewidth = 1.2)
plot(arange(0, len(reinforced_circ_avg_weights)*p.DTsim*p.samplingTime, p.DTsim*p.samplingTime), reinforced_circ_avg_weights, 'b-', linewidth = 1.2)
numTicks = 10
yticks( [ i * 1.0 /numTicks * p.Wmax for i in range(numTicks+1) ], [ "%.2f" % (i * 1.0/numTicks) for i in range(numTicks+1) ] )
ylim(0.45 * p.Wmax, 0.75*p.Wmax)            
xlim(0, p.Tsim+p.Tsim/1000)
xticks( array([ i * tickInterval for i in range(int(p.Tsim/tickInterval) + 1) ]), [ '%d' % (i * tickInterval / 60)  for i in range(int(p.Tsim/tickInterval) + 1) ])
ylabel('avg. weights $(w/w_{max})$')

xlabel('time [min]')
text(-0.30, 1.23, 'C', fontsize = 'x-large', transform = ax.transAxes )


ax = subplot(3, 2, 5, projection = 'frameaxes')
XBeforeMin = 3
XBeforeMax = 11

XAfterMin = 1184
XAfterMax = 1192

before_spikes = clip_window(r.spikes[p.reinforced_nrn_idx], XBeforeMin, XBeforeMax, shift = True)
after_spikes = clip_window(r.spikes[p.reinforced_nrn_idx], XAfterMin, XAfterMax, shift = True)
errorbar( before_spikes , 3*ones(len(before_spikes)), 0.6 * ones(len(before_spikes)), capsize = 0, visible = False, color = 'k')
errorbar( after_spikes , ones(len(after_spikes)), 0.6 * ones(len(after_spikes)), capsize = 0, visible = False, color = 'k')
xticks(arange(0,9,2), [ '%d' % (x,) for x in arange(0,9,2) ])
xlim(0, XBeforeMax - XBeforeMin)
xlabel("time [sec]")    
ylim(0,4)
yticks([])    

text(-0.22, 1.15, 'D', fontsize = 'x-large', transform = ax.transAxes )

text(-0.17, 0.75, 'before', horizontalalignment='center',
     verticalalignment='center', fontsize = 'medium', transform = ax.transAxes)
text(-0.17, 0.64, 'learning', horizontalalignment='center',
     verticalalignment='center', fontsize = 'medium', transform = ax.transAxes)

text(-0.17, 0.25, 'after', horizontalalignment='center',
     verticalalignment='center', fontsize = 'medium', transform = ax.transAxes)
text(-0.17, 0.14, 'learning', horizontalalignment='center',
     verticalalignment='center', fontsize = 'medium', transform = ax.transAxes)


ax = subplot(3,2,6, projection = 'frameaxes')
cross_corr = []
for i in range(len(r.exc_ou_afferents_reinforced_nrn)):        
    idx = r.exc_ou_nrn_idxs[r.exc_ou_afferents_reinforced_nrn[i]]
    cross_corr.append( cross_correlate_spikes(r.spikes[idx], r.spikes[p.reinforced_nrn_idx], 0.5e-3, -40e-3, 40e-3) )

for i in range(len(r.exc_other_afferents_reinforced_nrn)):        
    idx = r.exc_ou_nrn_idxs[r.exc_other_afferents_reinforced_nrn[i]]
    cross_corr.append( cross_correlate_spikes(r.spikes[idx], r.spikes[p.reinforced_nrn_idx], 0.5e-3, -40e-3, 40e-3) )

print "shape is ", cross_corr[0].shape
print "shape is ", vstack(tuple(cross_corr)).shape
avg_cross_corr = mean(vstack(tuple(cross_corr)), axis = 0)

plot(arange(0, 40.0e-3, 0.5e-3), avg_cross_corr[int(len(avg_cross_corr)/2):-1], 'k-', linewidth = 1.2)
plot(arange(0, 40.0e-3, 0.5e-3), avg_cross_corr[int(len(avg_cross_corr)/2):0:-1], 'r-', linewidth = 1.2)

ylabel('cross-correlation ($\cdot 10^{-3}$)', fontsize = 'medium')    
xlabel(' time [ms]')

xticks( arange(0, 40.5e-3, 10e-3), [ "%d" % (x,) for x in arange(0, 40.5e-3, 10e-3) * 1000] )
xlim(0,30.1e-3) 

text(-0.32, 1.15, 'E', fontsize = 'x-large', transform = ax.transAxes )
yticks(array([5,15,25])*10e-4,  [ '5', '15', '25' ] )

    
savefig("fetz_journal.eps")
