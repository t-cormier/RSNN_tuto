import os, sys
sys.path.append('../packages')
from pylab import *
from tables import *
import numpy
from math import *
from pypcsimplus import *
from pyV1.inputs import jitteredtemplate as jtempl
from frame import FrameAxes
from numpy import *

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('text', usetex=True)
rc('legend', fontsize=12)

# Load results

if len(sys.argv) > 1:
    h5filename = sys.argv[1]
else:
    h5filename = last_file(".*\.h5")
    
print " loading h5 filename : ", h5filename

h5file = openFile(h5filename, mode = "r", title = "Biofeedback DASTDP Experiment results")

p = constructParametersFromH5File(h5file)
r = constructRecordingsFromH5File(h5file)

ep = p.experiment

print p

rc = r.rewardInput
rc.spikes = rc.exc_spikes + rc.inh_spikes

r.input = r.rewardInput
p.input = p.rewardInput


A_p_theory = p.readout.DAStdpRate * p.readout.stdpApos * p.input.rewPulseScale / (ep.DTsim * 0.01 * p.readout.Wmax * p.input.rewTau * exp(1))

print "VALUE OF A_P_THEORY IS ", A_p_theory

display_fig = [30]

numpy.random.seed(123098)


utterShown = 9

startRewBin = p.experiment.initT
endRewBin = p.experiment.initT + p.input.rewardT + p.input.rewardDuration
 
# calculate the number of spikes in a bin (per template, per epoch)
epoch_learning_spikes = split_window(r.readout.learning_spikes, ep.trialT, len(r.SudList) * ep.trialT )
  
train_epoch_learning_spikes_len = [ [] for i in range(p.input.nDigits) ]
test_epoch_learning_spikes_len = [ [] for i in range(p.input.nDigits) ]
train_epoch_learning_spikes = [ [] for i in range(p.input.nDigits) ]
test_epoch_learning_spikes = [ [] for i in range(p.input.nDigits) ]

for epoch_i in range(r.sudListSegments[r.phaseNum['train']][1],r.sudListSegments[r.phaseNum['train']][2]):
    train_epoch_learning_spikes_len[ r.SudList[epoch_i][2] - 1 ].append(len(clip_window(epoch_learning_spikes[epoch_i], startRewBin, endRewBin)))
    train_epoch_learning_spikes[ r.SudList[epoch_i][2] - 1 ].append(clip_window(epoch_learning_spikes[epoch_i], startRewBin, endRewBin))
    
for epoch_i in range(r.sudListSegments[r.phaseNum['test']][1], r.sudListSegments[r.phaseNum['test']][2]):
    test_epoch_learning_spikes_len[ r.SudList[epoch_i][2] -1 ].append(len(clip_window(epoch_learning_spikes[epoch_i], startRewBin, endRewBin)))
    test_epoch_learning_spikes[ r.SudList[epoch_i][2] -1 ].append(clip_window(epoch_learning_spikes[epoch_i], startRewBin, endRewBin))
    
    

f = figure(30, figsize = (8,9), facecolor = 'w')
f.subplots_adjust(top= 0.94, left = 0.11, bottom = 0.1, right = 0.92, hspace = 0.76, wspace = 0.45)

numRandomChannels = 200

random_channels = random.permutation(len(rc.spikes))[:numRandomChannels]
random_chosen_response = [ rc.spikes[x] for x in random_channels ]

x_move = -0.19
x_space = 0.025
stretch = 2.0

stretch_main = 0.7

# plot liquid rc.spikes
################################################################     
ax = subplot(3, 3, 1)

ax_pos = ax.get_position().get_points().flatten()
ax_pos[2] -= ax_pos[0]
ax_pos[3] -= ax_pos[1]
   

leg_ax_pos = list(ax_pos)
leg_ax_pos[2] = stretch * leg_ax_pos[2] * stretch_main
ax.set_position(leg_ax_pos)


leftX = r.sudListSegments[r.phaseNum['preTrain']][1]*ep.trialT; rightX = leftX + ep.trialT
raster_x, raster_y = create_raster(random_chosen_response, leftX, rightX)
orig_raster_x, orig_raster_y = raster_x, raster_y
raster_x -= leftX

xlabel('time [ms]')

ylabel('200 neurons')


mark_size = 6
rect_color = '0.92'

plot(raster_x, raster_y, 'r.', markersize = mark_size, color = 'k')

ylim(0,numRandomChannels)
yticks(arange(0,numRandomChannels,100), [ '' for x in arange(0,numRandomChannels,100) ] )
axvline(0.2, color = 'k', linestyle = ':')
axvline(0.3, color = 'k', linestyle = ':')
xticks(arange(0.100,0.501,0.1), [ '%d' % (x,) for x in arange(0,401,100)])
xlim(0.1,0.5)

text(-0.16, 1.08, 'A', fontsize = 'x-large', transform = ax.transAxes)

rect = Rectangle((0.25,0.0), 0.25, 1.0, fill = True, facecolor = rect_color, edgecolor = rect_color, transform = ax.transAxes)    
ax.add_patch(rect)    

#################################################################
#second axes
main_ax = subplot(3,3,2)
main_ax.set_visible(False)

ax_pos = main_ax.get_position().get_points().flatten()
ax_pos[2] -= ax_pos[0]
ax_pos[3] -= ax_pos[1]

leg_ax_pos = list(ax_pos)    
leg_ax_pos[0] += (x_move + f.subplotpars.wspace) * stretch - f.subplotpars.wspace 
leg_ax_pos[2] *= 0.25 * stretch      
ax = axes(leg_ax_pos)


leftX = (r.sudListSegments[r.phaseNum['preTrain']][1]+10)*ep.trialT; rightX = leftX + ep.trialT
raster_x, raster_y = create_raster(random_chosen_response, leftX, rightX)
raster_x -= leftX
plot(raster_x, raster_y, 'b.', markersize = mark_size, color = 'r')
plot(orig_raster_x, orig_raster_y, 'r.', markersize = mark_size, color  = 'k')


ylim(0,numRandomChannels)
yticks([])


xlim(0.2,0.301)    
xticks(arange(0.2,0.301,0.1), [ '%d' % (x,) for x in arange(100,201,100)])

text(-0.16, 1.08, 'B', fontsize = 'x-large', transform = ax.transAxes)



###############################################################
# third axes

ax_pos = main_ax.get_position().get_points().flatten()
ax_pos[2] -= ax_pos[0]
ax_pos[3] -= ax_pos[1]


leg_ax_pos = list(ax_pos)
leg_ax_pos[0] += (x_space + 0.25 * leg_ax_pos[2])*stretch + (x_move + f.subplotpars.wspace) * stretch - f.subplotpars.wspace 
leg_ax_pos[2] *= 0.25 * stretch      
ax = axes(leg_ax_pos)

# plot liquid rc.spikes

leftX = (r.sudListSegments[r.phaseNum['preTrain']][1]+1)*ep.trialT; rightX = leftX + ep.trialT
raster_x, raster_y = create_raster(random_chosen_response, leftX, rightX)
raster_x -= leftX
plot(raster_x, raster_y, 'b.', markersize = mark_size, color = 'r')
plot(orig_raster_x, orig_raster_y, 'r.', markersize = mark_size, color  = 'k')

ylim(0,numRandomChannels)
yticks([])

xlim(0.2,0.301)
xticks(arange(0.2,0.301,0.1), [ '' for x in arange(100,301,50)])
xlabel('time [ms]')
xticks(arange(0.2,0.301,0.1), [ '%d' % (x,) for x in arange(100,201,100)])


text(-0.16, 1.08, 'C', fontsize = 'x-large', transform = ax.transAxes)
#################################################################
# fourth


ax_pos = main_ax.get_position().get_points().flatten()
ax_pos[2] -= ax_pos[0]
ax_pos[3] -= ax_pos[1]


leg_ax_pos = list(ax_pos)
leg_ax_pos[0] += (2 * (x_space + 0.25 * leg_ax_pos[2]))*stretch + (x_move + f.subplotpars.wspace) * stretch - f.subplotpars.wspace 
leg_ax_pos[2] *= 0.25 * stretch      
ax = axes(leg_ax_pos)



leftX = (r.sudListSegments[r.phaseNum['preTrain']][1]+20)*ep.trialT; rightX = leftX + ep.trialT
raster_x, raster_y = create_raster(random_chosen_response, leftX, rightX)
raster_x -= leftX
plot(raster_x, raster_y, 'b.', markersize = mark_size, color = 'r')
plot(orig_raster_x, orig_raster_y, 'r.', markersize = mark_size, color  = 'k')



ylim(0,numRandomChannels)
yticks([])


xlim(0.2,0.301)

xticks(arange(0.2,0.301,0.1), [ '%d' % (x,) for x in arange(100,201,100)])    

text(-0.16, 1.08, 'D', fontsize = 'x-large', transform = ax.transAxes)

skipTrials = 4
label_fig = ['F', 'E']
digit_str = [ '"one"', '"two"' ]
nTrials = ep.nTrainEpochs
for tmpl_i in range(p.input.nDigits):
    ax = subplot(3, 3, 4 + (tmpl_i + 1) % 2)
    if tmpl_i == 0:
        ax_pos = ax.get_position().get_points().flatten()
        ax_pos[2] -= ax_pos[0]
        ax_pos[3] -= ax_pos[1]
        
        leg_ax_pos = list(ax_pos)
        leg_ax_pos[0] -= 0.2 *leg_ax_pos[2]
        ax.set_position(leg_ax_pos)
        
    raster_x, raster_y = create_raster(train_epoch_learning_spikes[tmpl_i][::skipTrials], 0, ep.trialT)
    orig_raster_x = raster_x
    orig_raster_y = raster_y
    raster_x -= ep.initT        
    plot(raster_x, raster_y, 'k.')
    
    xticks(arange(0,0.501,0.1), [ '%d' % (x,) for x in arange(0,501,100)])
    xlim(0, 0.4)
    
    ylim(0,int(nTrials/2.0/skipTrials))
    yticks( arange(0,max(raster_y)+1,max(raster_y)/4), [ '%d' % (x,) for x in arange(0,nTrials/2 + 0.1,250) ])        
    text(0.5, 1.18, 'readout response', horizontalalignment = 'center', verticalalignment = 'center', 
        fontsize = 'medium', transform = ax.transAxes)
    title('to digit ' + digit_str[tmpl_i] , fontsize = 'medium')
    if tmpl_i == 1:
        ylabel('trial \#')
    else:
        yticks(arange(0,max(raster_y)+1,max(raster_y)/2),[])
    xlabel('time [ms]')
    
text(-0.24,1.1,'E', fontsize = 'x-large', transform = ax.transAxes)



ax = subplot(3, 3, 6, projection = 'frameaxes')

labels_plot = [ 'digit "one"', 'digit "two"']
colors_plot = [ 'b', 'g']
moving_average = 40.0
for templ_i in range(p.input.nDigits):        
    plot(hstack((zeros(moving_average), convolve(train_epoch_learning_spikes_len[templ_i],ones(moving_average), mode = 'valid')/moving_average)), color = colors_plot[templ_i], 
        label = labels_plot[templ_i], linewidth = 1.2)


ylabel('num. of readout spikes')

xticks(arange(0,nTrials/2+1,250), [ '%d' % (x,) for x in arange(0,nTrials+1,500) ])
xlim(moving_average,nTrials/2 + 1)
xlabel('trial \#')
text(-0.24,1.08, 'F', fontsize = 'x-large', transform = ax.transAxes)    
yticks(arange(0,5.1,1), [ '%d' % (x,) for x in arange(0,5.1,1)])
ylim(0,4.4)

    
legend(loc = (0.6,0.25), markerscale = 2.0)

#vm before after learning without threshold -> positive pattern
ax = subplot(3,2,6, projection = 'frameaxes')
before_positive_no_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, utterShown*ep.trialT + ep.initT, utterShown*ep.trialT + ep.initT + ep.liq_input.templDuration)    
after_positive_no_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, (60+utterShown)*ep.trialT + ep.initT, (60+utterShown)*ep.trialT + ep.initT + ep.liq_input.templDuration)
plot(arange(0,len(before_positive_no_thresh)*ep.DTsim, ep.DTsim), before_positive_no_thresh, 'b-')
plot(arange(0,len(after_positive_no_thresh)*ep.DTsim, ep.DTsim), after_positive_no_thresh, 'r-')

print "variance before learning negative => ", std(before_positive_no_thresh)**2
print "variance after learning negative  => ", std(after_positive_no_thresh)**2

#xticks([])
xticks(arange(0,0.51,0.1), [])    
# xlabel('time [ms]')
yticks( arange(-0.067,-0.0549,0.002), [ '%d' % (x) for x in arange(-67,-54.5,2) ])
ylabel('$V_{m}(t)$ [mV]', fontsize = 'smaller')

ylim(-0.0672,-0.054)
text(-0.19,1.08,'H',fontsize = 'x-large', transform = ax.transAxes)

title('response to utterance of digit "one"', fontsize = 'medium')

xticks(arange(0,0.51,0.1), [ '%d' % x for x in arange(0,501,100) ] )    
xlabel('time [ms]')
xlim(0,0.401)

    

ax = subplot(3,2,5, projection = 'frameaxes')
before_negative_no_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, (10+utterShown)*ep.trialT + ep.initT, (10+utterShown)*ep.trialT + ep.initT + ep.liq_input.templDuration)    
after_negative_no_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim,  (70+utterShown)*ep.trialT + ep.initT, (70+utterShown)*ep.trialT + ep.initT + ep.liq_input.templDuration)
    
plot(arange(0,len(before_negative_no_thresh)*ep.DTsim, ep.DTsim), before_negative_no_thresh, 'b-')
plot(arange(0,len(after_negative_no_thresh)*ep.DTsim, ep.DTsim), after_negative_no_thresh, 'r-')

print "variance before learning positive  => ", std(before_negative_no_thresh)**2
print "variance after learning positive  => ", std(after_negative_no_thresh)**2


xticks(arange(0,0.51,0.1), [])    

yticks( arange(-0.067,-0.050,0.002), [ '%d' % (x) for x in arange(-67,-50,2) ])
ylabel('$V_{m}(t)$ [mV]', fontsize = 'smaller')

text(-0.19,1.08,'G',fontsize = 'x-large', transform = ax.transAxes)
ylim(-0.0672,-0.050)
title('response to utterance of digit "two"', fontsize = 'medium')


xticks(arange(0,0.51,0.1), [ '%d' % x for x in arange(0,501,100) ] )
xlim(0,0.401)    
xlabel('time [ms]')

savefig('speech_fig.eps')
