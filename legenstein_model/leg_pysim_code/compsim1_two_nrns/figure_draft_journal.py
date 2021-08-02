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


if len(sys.argv) > 1:
    h5filename = sys.argv[1]
else:
    h5filename = last_file(".*\.h5")
    
print " loading h5 filename : ", h5filename

h5file = openFile(h5filename, mode = "r", title = "Biofeedback DASTDP Experiment results")

p = constructParametersFromH5File(h5file)
r = constructRecordingsFromH5File(h5file)

print p


other_circ_not_ou_weights = r.other_circ_not_ou_weights.copy()
other_circ_ou_weights = r.other_circ_ou_weights.copy()

other_circ_ou_weights *= 1.0/p.WHighOUScale
other_circ_not_ou_weights *= 1.0/p.WLowOUScale


other_weights = vstack((other_circ_not_ou_weights, other_circ_ou_weights))

reinforced_circ_weights = [1,2]
r.reinforced_ou_weights = [1,2]
r.reinforced_other_weights = [1,2]

r.reinforced_ou_weights[0] = getattr(r, "reinforced_ou_weights_0")
r.reinforced_ou_weights[1] = getattr(r, "reinforced_ou_weights_1")

r.reinforced_other_weights[0] = getattr(r, "reinforced_other_weights_0")
r.reinforced_other_weights[1] = getattr(r, "reinforced_other_weights_1")

reinforced_circ_weights = [1,2]
for i in range(2):
    reinforced_circ_weights[i] = vstack((r.reinforced_ou_weights[i], r.reinforced_other_weights[i]))


reinforced_circ_weights *= 1.0/p.WLowOUScale

exc_ou_spikes = []
for i in r.exc_ou_nrn_idxs:
    exc_ou_spikes.append(r.spikes[i])
    
exc_other_spikes = []
for i in r.exc_other_nrn_idxs:
    exc_other_spikes.append(r.spikes[i])

reinforced_circ_avg_weights = [1,2]
reinforced_circ_std_weights = [1,2]

reinforced_ou_avg_weights = [1,2]
reinforced_ou_std_weights = [1,2]

reinforced_other_avg_weights = [1,2]
reinforced_other_std_weights = [1,2]


for i in range(2):
    reinforced_circ_avg_weights[i] = average(reinforced_circ_weights[i], 0)
    reinforced_circ_std_weights[i] = std(reinforced_circ_weights[i], 0)

    reinforced_ou_avg_weights[i] = average(r.reinforced_ou_weights[i], 0)
    reinforced_ou_std_weights[i] = std(r.reinforced_ou_weights[i], 0)

    reinforced_other_avg_weights[i] = average(r.reinforced_other_weights[i], 0)
    reinforced_other_std_weights[i] = std(r.reinforced_other_weights[i], 0)

other_avg_weights = average(other_weights, 0)
other_std_weights = std(other_weights, 0)


other_circ_avg_not_ou_weights = average(other_circ_not_ou_weights, 0)
other_circ_std_not_ou_weights = std(other_circ_not_ou_weights, 0)

other_circ_avg_ou_weights = average(other_circ_ou_weights, 0)
other_circ_std_ou_weights = std(other_circ_ou_weights, 0)


f = figure(1,figsize=(8,6), facecolor = 'w')

f.subplots_adjust(top= 0.93, left = 0.1, bottom = 0.10, right = 0.95, hspace = 0.76, wspace = 0.54)
clf()

# plot liquid r.spikes
nRespNeurons = 99
leftX = 20; rightX = 23 
gap_duration = 0.5

numpy.random.seed(34223515)

chosenNeuronsArray = numpy.random.permutation(p.nNeurons)[:nRespNeurons]
chosenNeuronsArraySplit1 = chosenNeuronsArray[:int(nRespNeurons / 3)]
chosenNeuronsArraySplit2 = chosenNeuronsArray[int(nRespNeurons / 3):int(2 * nRespNeurons / 3)]
chosenNeuronsArraySplit3 = chosenNeuronsArray[int(2*nRespNeurons / 3):]
chosenNeuronsArray = hstack((chosenNeuronsArraySplit1,p.reinforced_nrn_idx[0],chosenNeuronsArraySplit2,p.reinforced_nrn_idx[1],chosenNeuronsArraySplit3))

chosenSpikes = [ r.spikes[i] for i in chosenNeuronsArray ]
reinfNeuronRasterIdx = [1,2]
reinfNeuronRasterIdx[0] = int(nRespNeurons / 3)
reinfNeuronRasterIdx[1] = int(2 * nRespNeurons / 3) + 1

clipped_spikes_reinforc = [1, 2]
raster_x, raster_y = create_raster(chosenSpikes, leftX, rightX, shift = True)        
clipped_spikes_reinforc[0] = clip_window(chosenSpikes[reinfNeuronRasterIdx[0]], leftX, rightX, shift = True)
clipped_spikes_reinforc[1] = clip_window(chosenSpikes[reinfNeuronRasterIdx[1]], leftX, rightX, shift = True)

leftX_2, rightX_2 = p.Tsim/2  - 3, p.Tsim /2
raster_x_2, raster_y_2 = create_raster(chosenSpikes,leftX_2, rightX_2, shift = True)
raster_x_2 += (rightX - leftX) + gap_duration
clipped_spikes_reinforc_2 = [1,2]
clipped_spikes_reinforc_2[0] = clip_window(chosenSpikes[reinfNeuronRasterIdx[0]], leftX_2, rightX_2, shift = True) + (rightX - leftX) + gap_duration 
clipped_spikes_reinforc_2[1] = clip_window(chosenSpikes[reinfNeuronRasterIdx[1]], leftX_2, rightX_2, shift = True) + (rightX - leftX) + gap_duration


total_raster_x = hstack((raster_x, raster_x_2))
total_raster_y = hstack((raster_y, raster_y_2))

total_clip_spikes_reinforc = [1,2]

total_clip_spikes_reinforc[0] = hstack((clipped_spikes_reinforc[0], clipped_spikes_reinforc_2[0]))
total_clip_spikes_reinforc[1] = hstack((clipped_spikes_reinforc[1], clipped_spikes_reinforc_2[1]))
 
ax = subplot(2, 1, 1, projection = 'frameaxes')
plot(total_raster_x, total_raster_y, '.', color = '0.0', markersize = 0.3)
plot(total_clip_spikes_reinforc[0], [ reinfNeuronRasterIdx[0] for i in range(len(total_clip_spikes_reinforc[0])) ], '+', color = '0.0', markersize = 10, mec = 'b', mfc = 'b',  markeredgewidth = 2)
plot(total_clip_spikes_reinforc[1], [ reinfNeuronRasterIdx[1] for i in range(len(total_clip_spikes_reinforc[1])) ], '+', color = '0.0', markersize = 10, mec = 'g', mfc = 'g',  markeredgewidth = 2)

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
    
    

ax = subplot(2, 2, 3, projection = 'frameaxes')
reinforced_nrn_rate = []
reinforced_nrn_rate.append(calc_rate_2(r.spikes[p.reinforced_nrn_idx[0]], int(p.Tsim/(p.samplingTime * p.DTsim / 4)), 40, p.Tsim))
reinforced_nrn_rate.append(calc_rate_2(r.spikes[p.reinforced_nrn_idx[1]], int(p.Tsim/(p.samplingTime * p.DTsim / 4)), 40, p.Tsim))

nrn_rate = []
for s in r.spikes[10:30]:
    nrn_rate.append(calc_rate_2(s, int(p.Tsim/(p.samplingTime * p.DTsim / 4)), 40, p.Tsim))
avg_nrn_rate = []
for i in range(len(nrn_rate[0])):
    avg_nrn_rate.append(sum([ nrn_rate[k][i] for k in range(len(nrn_rate)) ]) / len(nrn_rate))    
plot(arange(0, (len(reinforced_nrn_rate[0]) - .5) * p.DTsim * p.samplingTime / 4, p.DTsim * p.samplingTime / 4), reinforced_nrn_rate[0], 'b-', linewidth = 1.2)
plot(arange(0, (len(reinforced_nrn_rate[1]) - .5) * p.DTsim * p.samplingTime / 4, p.DTsim * p.samplingTime / 4), reinforced_nrn_rate[1], 'g-', linewidth = 1.2)
plot(arange(0, (len(avg_nrn_rate) - .5) * p.DTsim * p.samplingTime / 4, p.DTsim * p.samplingTime / 4), avg_nrn_rate, 'k--', linewidth = 1.2)
theGap = 10 * p.DTsim * p.samplingTime + 10 * p.DTsim * p.samplingTime / 4
print "theGap is", theGap
xlim(0, p.Tsim - theGap + p.Tsim/1000)
tickInterval = 300

xticks( array([ i * tickInterval - theGap for i in range(1, int(p.Tsim/tickInterval) + 1) ]), [ '%d' % (i * tickInterval / 60)  for i in range(1, int(p.Tsim/tickInterval) + 1) ])
xlabel("time [min]")
ylim(0,16.1)
yticks(arange(0,16.1,4), [ '%d' % (x,) for x in arange(0,16.1,4) ])
ylabel("rate [Hz]")
axvline(p.Tsim/2-theGap, linestyle = ':', color = 'k')    
text(-0.24, 1.15, 'B', fontsize = 'x-large', transform = ax.transAxes )


reinf_fontsize = 14
text(0.10, 1.0, 'A$\uparrow$', color = 'b', fontsize = reinf_fontsize, transform = ax.transAxes)
text(0.20, 1.01, '+', color = 'k', fontsize = reinf_fontsize - 1, transform = ax.transAxes)
text(0.28, 1.0,'B$\downarrow$', color = 'g', fontsize = reinf_fontsize, transform = ax.transAxes )

text(0.60, 1.0, 'A$\downarrow$', color = 'b', fontsize = reinf_fontsize, transform = ax.transAxes)
text(0.70, 1.01, '+', color = 'k', fontsize = reinf_fontsize - 1, transform = ax.transAxes)
text(0.78, 1.0,'B$\uparrow$', color = 'g', fontsize = reinf_fontsize, transform = ax.transAxes )



ax = subplot(2, 2, 4, projection = 'frameaxes')
plot(arange(0, (len(other_avg_weights) -0.5) * p.DTsim*p.samplingTime, p.DTsim*p.samplingTime), other_avg_weights, 'k--', linewidth = 1.2)

plot(arange(0, len(reinforced_circ_avg_weights[0])*p.DTsim*p.samplingTime, p.DTsim*p.samplingTime), reinforced_circ_avg_weights[0], 'b-', linewidth = 1.2)
plot(arange(0, len(reinforced_circ_avg_weights[1])*p.DTsim*p.samplingTime, p.DTsim*p.samplingTime), reinforced_circ_avg_weights[1], 'g-', linewidth = 1.2)

p.Wmax = 2 * p.WexcLowOU

numTicks = 10

yticks( [ i * 1.0 /numTicks * p.Wmax for i in range(numTicks+1) ], [ "%.1f" % (i * 1.0/numTicks) for i in range(numTicks+1) ] )
ylim(0.30 * p.Wmax, 0.70*p.Wmax)
axvline(p.Tsim/2, linestyle = ':' , color = 'k')            
xlim(0, p.Tsim+p.Tsim/1000)    
xticks( array([ i * tickInterval for i in range(int(p.Tsim/tickInterval) + 1) ]), [ '%d' % (i * tickInterval / 60)  for i in range(int(p.Tsim/tickInterval) + 1) ])
ylabel('avg. weights $(w/w_{max})$')

xlabel('time [min]')
text(-0.30, 1.23, 'C', fontsize = 'x-large', transform = ax.transAxes )

text(0.11, 1.0, 'A$\uparrow$', color = 'b', fontsize = reinf_fontsize, transform = ax.transAxes)
text(0.21, 1.01, '+', color = 'k', fontsize = reinf_fontsize - 1, transform = ax.transAxes)
text(0.29, 1.0,'B$\downarrow$', color = 'g', fontsize = reinf_fontsize, transform = ax.transAxes )

text(0.61, 1.0, 'A$\downarrow$', color = 'b', fontsize = reinf_fontsize, transform = ax.transAxes)
text(0.71, 1.01, '+', color = 'k', fontsize = reinf_fontsize - 1, transform = ax.transAxes)
text(0.79, 1.0,'B$\uparrow$', color = 'g', fontsize = reinf_fontsize, transform = ax.transAxes )

    
savefig("fetz_two_nrns.eps")

