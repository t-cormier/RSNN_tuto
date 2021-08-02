import os, sys
from pylab import *
from tables import *
from numpy import *
import numpy
from math import *
sys.path.append('../packages')
from pyV1.inputs import jitteredtemplate as jtempl
from pypcsimplus import *
from matplotlib import rc as resources
from frame import FrameAxes
import re
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('text', usetex=True)

if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    directory = last_created_dir(".*")

filenames = os.listdir(directory)
patt = re.compile(".*\.h5")
filenames = [ os.path.join(directory, x) for x in filenames if patt.match(x) ]

filenames.sort()

recs = []
pars = []
epars = []
rc = []

shown_exp = 0

for f in filenames:
    recs.append(constructRecordingsFromH5File(f))
    pars.append(constructParametersFromH5File(f))
    epars.append( pars[-1].experiment )
             

total_num_exp = len(filenames)

h5filename = filenames[shown_exp]
    
print " loading h5 filename : ", h5filename

h5file = openFile(h5filename, mode = "r", title = "Biofeedback DASTDP Experiment results")

p = constructParametersFromH5File(h5file)
r = constructRecordingsFromH5File(h5file)
ep = p.experiment

print p

template_pre_test_epoch_learning_spikes = []
template_train_epoch_learning_spikes = []
template_test_epoch_learning_spikes = []


pre_test_mean_spikes = [ [] for i in range(pars[0].input.nTemplates) ] 
test_mean_spikes = [ [] for i in range(pars[0].input.nTemplates) ]
pre_test_std_spikes = [ [] for i in range(pars[0].input.nTemplates) ]
test_std_spikes = [ [] for i in range(pars[0].input.nTemplates) ]

for exp_num in range(len(filenames)):

    startRewBin = pars[exp_num].input.initT
    endRewBin = pars[exp_num].input.initT + pars[exp_num].input.rewardT + pars[exp_num].input.rewardDuration
     
    # calculate the number of spikes in a bin (per template, per epoch)    
    epoch_learning_spikes = split_window(recs[exp_num].readout.learning_spikes, epars[exp_num].trialT)
    
    while len(epoch_learning_spikes) < (pars[exp_num].experiment.nTrainEpochs + 2*pars[exp_num].experiment.nTestEpochs):
        epoch_learning_spikes.append(array([]))
        
    if exp_num == shown_exp:
        epoch_learning_spikes_shown = epoch_learning_spikes
      
    template_pre_test_epoch_learning_spikes.append( [ [] for i in range(pars[exp_num].input.nTemplates) ] )
    template_train_epoch_learning_spikes.append( [ [] for i in range(pars[exp_num].input.nTemplates) ] )
    template_test_epoch_learning_spikes.append( [ [] for i in range(pars[exp_num].input.nTemplates) ] )
    
    for epoch_i in range(10,epars[exp_num].nTestEpochs):
        template_pre_test_epoch_learning_spikes[exp_num][ recs[exp_num].input.chosenTemplates[epoch_i] ].append(len(clip_window(epoch_learning_spikes[epoch_i], startRewBin, endRewBin)))
    
    for epoch_i in range(epars[exp_num].nTestEpochs, epars[exp_num].nTestEpochs + epars[exp_num].nTrainEpochs):
        template_train_epoch_learning_spikes[exp_num][ recs[exp_num].input.chosenTemplates[epoch_i] ].append(len(clip_window(epoch_learning_spikes[epoch_i], startRewBin, endRewBin)))
        
    for epoch_i in range(2+epars[exp_num].nTrainEpochs+epars[exp_num].nTestEpochs, epars[exp_num].nTrainEpochs + 2*epars[exp_num].nTestEpochs-10):
         template_test_epoch_learning_spikes[exp_num][ recs[exp_num].input.chosenTemplates[epoch_i] ].append(len(clip_window(epoch_learning_spikes[epoch_i], startRewBin, endRewBin)))
         
    pre_test_mean_spikes.append([])
    test_mean_spikes.append([])
    pre_test_std_spikes.append([])
    test_std_spikes.append([])
    
    for tmpl_i in range(pars[exp_num].input.nTemplates):            
        pre_test_mean_spikes[tmpl_i].append(mean(template_pre_test_epoch_learning_spikes[exp_num][tmpl_i]))
        test_mean_spikes[tmpl_i].append(mean(template_test_epoch_learning_spikes[exp_num][tmpl_i]))            
        pre_test_std_spikes[tmpl_i].append(std(template_pre_test_epoch_learning_spikes[exp_num][tmpl_i])) 
        test_std_spikes[tmpl_i].append(std(template_test_epoch_learning_spikes[exp_num][tmpl_i]))

resources('xtick.major', size=4)
f = figure(4, figsize = (8,10.5), facecolor = 'w')
clf()

f.subplots_adjust(top= 0.94, left = 0.11, bottom = 0.06, right = 0.97, hspace = 0.65, wspace = 0.55)

stepTrial = 4

totalNumTrials = float(ep.nTrainEpochs + 2*ep.nTestEpochs - 2*ep.numTrialsWithoutThreshold)
print "totalNumTrials = ", totalNumTrials

panel_letters = ['A', 'B']
displayed_range = [1,2]
for tmpl_i in range(p.input.nTemplates):        
    ax = subplot(4, 2, 1 + tmpl_i)    
    raster_x, raster_y = create_raster(epoch_learning_spikes_shown[ep.numTrialsWithoutThreshold + 2 + 2*int(p.input.targetTemplate == tmpl_i)*stepTrial + tmpl_i:2*ep.nTestEpochs+ep.nTrainEpochs-ep.numTrialsWithoutThreshold:stepTrial*p.input.nTemplates], 0, ep.trialT)
    displayed_range[tmpl_i] = range(ep.numTrialsWithoutThreshold + tmpl_i+2,2*ep.nTestEpochs+ep.nTrainEpochs-ep.numTrialsWithoutThreshold, stepTrial*p.input.nTemplates)
    vlines(raster_x, raster_y, raster_y + 2.0, color = 'k', linestyle = '-', linewidth=1.3)
    
            
    ylim(0,max(raster_y) + 1)
    xticks(arange(0+p.input.initT,0.51+0+p.input.initT,0.1), [])
    xlim(p.input.initT,p.input.initT + p.input.templDuration+0.001)    
    
    len_trials = len(epoch_learning_spikes_shown[ep.numTrialsWithoutThreshold+tmpl_i:2*ep.nTestEpochs+ep.nTrainEpochs-ep.numTrialsWithoutThreshold:stepTrial*p.input.nTemplates])
    yticks(arange(0, len_trials+0.001, float(len_trials)/(totalNumTrials/200)*2),\
           [ '%d' % (x) for x in arange(0, float(totalNumTrials)/2+1, 200) ] )
    ylabel('trial \#')
    if tmpl_i == p.input.targetTemplate:
        title('response to pattern P', fontsize = 'medium')
    else:
        title('response to pattern N', fontsize = 'medium')
    text(-0.29,1.08, panel_letters[tmpl_i],fontsize = 'x-large', transform = ax.transAxes)
    
    
#vm before after learning without threshold -> positive pattern
ax = subplot(4,2,3, projection = 'frameaxes')
before_positive_no_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, 2*ep.trialT + p.input.initT, 2*ep.trialT + p.input.initT + p.input.templDuration)    
after_positive_no_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, 36*ep.trialT + p.input.initT, 36*ep.trialT + p.input.initT + p.input.templDuration)
plot(arange(0,len(before_positive_no_thresh)*ep.DTsim, ep.DTsim), before_positive_no_thresh, 'b-')
plot(arange(0,len(after_positive_no_thresh)*ep.DTsim, ep.DTsim), after_positive_no_thresh, 'r-')

print "variance before learning positive => ", std(before_positive_no_thresh)**2
print "variance after learning positive  => ", std(after_positive_no_thresh)**2


xticks(arange(0,0.51,0.1), [])    

yticks( arange(-0.068,-0.0559,0.004), [ '%d' % (x) for x in arange(-68,-55.5,4) ])
ylabel('$V_{m}(t)$ [mV]', fontsize = 'smaller')

ylim(-0.068,-0.0559)
text(-0.29,1.08,'C',fontsize = 'x-large', transform = ax.transAxes)

title('response to pattern P', fontsize = 'medium')

ax = subplot(4,2,4, projection = 'frameaxes')
before_negative_no_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, 3*ep.trialT + p.input.initT, 3*ep.trialT + p.input.initT + p.input.templDuration)    
after_negative_no_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, 33*ep.trialT + p.input.initT, 33*ep.trialT + p.input.initT + p.input.templDuration)
    
plot(arange(0,len(before_negative_no_thresh)*ep.DTsim, ep.DTsim), before_negative_no_thresh, 'b-')
plot(arange(0,len(after_negative_no_thresh)*ep.DTsim, ep.DTsim), after_negative_no_thresh, 'r-')

print "variance after learning negative  => ", std(before_negative_no_thresh)**2
print "variance after learning negative  => ", std(after_negative_no_thresh)**2


xticks(arange(0,0.51,0.1), [])    

yticks( arange(-0.068,-0.0559,0.004), [ '%d' % (x) for x in arange(-68,-55.5,4) ])
ylabel('$V_{m}(t)$ [mV]', fontsize = 'smaller')

text(-0.29,1.08,'D',fontsize = 'x-large', transform = ax.transAxes)
ylim(-0.066,-0.0559)
title('response to pattern N', fontsize = 'medium')

ax = subplot(4,2,5, projection = 'frameaxes')
before_positive_with_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, 
                 (displayed_range[p.input.targetTemplate][1])*ep.trialT + p.input.initT, (displayed_range[p.input.targetTemplate][1])*ep.trialT + p.input.initT + p.input.templDuration)

print "after with threshold ", displayed_range[p.input.targetTemplate][-1]
    
trial_after_with_thresh = ep.numTrialsRecordVm + 1 + (displayed_range[p.input.targetTemplate][-1] - (2*ep.nTestEpochs + ep.nTrainEpochs - (ep.numTrialsRecordVm-1)))         
after_positive_with_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim,
               trial_after_with_thresh*ep.trialT + p.input.initT, trial_after_with_thresh*ep.trialT + p.input.initT + p.input.templDuration)

for x in epoch_learning_spikes_shown[displayed_range[p.input.targetTemplate][1]] - p.input.initT:
    axvline(x, color = 'b')
for x in epoch_learning_spikes_shown[displayed_range[p.input.targetTemplate][-1]] - p.input.initT: 
    axvline(x, color = 'r')    
plot(arange(0,len(before_positive_with_thresh)*ep.DTsim, ep.DTsim), before_positive_with_thresh, 'b-')
plot(arange(0,len(after_positive_with_thresh)*ep.DTsim, ep.DTsim), after_positive_with_thresh, 'r-', linewidth = 1.2)

xticks(arange(0,0.51,0.1), [ '%d' % x for x in arange(0,501,100) ] )    
xlabel('time [ms]')

axhline(-0.059, color = 'k', ls = '--')
yticks( arange(-0.072,-0.0575,0.004), [ '%d' % (x) for x in arange(-72,57.5,4) ])
ylabel('$V_{m}(t)$ [mV]', fontsize = 'smaller')

text(-0.29,1.08,'E',fontsize = 'x-large', transform = ax.transAxes)
ylim(-0.072,-0.057)
title('response to pattern P', fontsize = 'medium')


ax = subplot(4,2,6, projection = 'frameaxes')
p.input.otherTemplate = (p.input.targetTemplate + 1) % 2
trial_after_with_thresh = ep.numTrialsRecordVm + 1 + (displayed_range[p.input.otherTemplate][-1] - (2*ep.nTestEpochs + ep.nTrainEpochs -(ep.numTrialsRecordVm-1)))    
before_negative_with_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, (displayed_range[p.input.otherTemplate][0])*ep.trialT + p.input.initT, (displayed_range[p.input.otherTemplate][0])*ep.trialT + p.input.initT + p.input.templDuration)    
after_negative_with_thresh = clip_window_analog(r.readout.learning_nrn_vm, ep.DTsim, trial_after_with_thresh*ep.trialT + p.input.initT, trial_after_with_thresh*ep.trialT + p.input.initT + p.input.templDuration)
for x in epoch_learning_spikes_shown[displayed_range[p.input.otherTemplate][0]] - p.input.initT:
    axvline(x, color = 'b')
for x in epoch_learning_spikes_shown[displayed_range[p.input.otherTemplate][-1]] - p.input.initT: 
    axvline(x, color = 'r') 



plot(arange(0,len(before_negative_with_thresh)*ep.DTsim, ep.DTsim), before_negative_with_thresh, 'b-')
plot(arange(0,len(after_negative_with_thresh)*ep.DTsim, ep.DTsim), after_negative_with_thresh, 'r-', linewidth = 1.2)

xticks(arange(0,0.51,0.1), [ '%d' % x for x in arange(0,501,100) ] )    
xlabel('time [ms]')
axhline(-0.059, color = 'k', ls = '--')

ylim(-0.072,-0.057)
yticks( arange(-0.072,-0.0579,0.004), [ '%d' % (x) for x in arange(-72,-57.5,4) ])    
text(-0.29,1.08,'F',fontsize = 'x-large', transform = ax.transAxes)
ylabel('$V_{m}(t)$ [mV]', fontsize = 'smaller')

title('response to pattern N', fontsize = 'medium')

    
xticks(arange(0,0.51,0.1), [ '%d' % x for x in arange(0,501,100) ] )

        
#vm before after learning with threshold -> negative pattern


resources('xtick.major', size=0)

ax = subplot(4,2,7, projection = 'frameaxes')
    
bar( arange(30, 100*total_num_exp, 100), pre_test_mean_spikes[0], width = 15, yerr = pre_test_std_spikes[0], ecolor = 'k', color = '0.9' )
bar( arange(55, 100*total_num_exp, 100), pre_test_mean_spikes[1], width = 15, yerr = pre_test_std_spikes[1], ecolor = 'k', color = 'k')



text(-0.24, 0.5, 'num. of spikes', fontsize = 'smaller', horizontalalignment = 'center', 
         verticalalignment = 'center', rotation = 90, transform = ax.transAxes)
text(-0.16, 0.5, 'per trial', fontsize = 'smaller', horizontalalignment = 'center', 
         verticalalignment = 'center', rotation = 90, transform = ax.transAxes)

text(0.5, 0.75, 'before learning', fontsize = 'smaller', horizontalalignment = 'left', 
         verticalalignment = 'bottom', transform = ax.transAxes)

xlabel('experiment \#')

yticks(arange(0,6.5,2), [ '%d' % (x,) for x in arange(0,6.5,2) ])
ylim(0,8.1)


xticks(arange(50, 100*total_num_exp, 100), [ '%d' % (x+1,) for x in range(total_num_exp) ])

text(-0.29,1.08,'G',fontsize = 'x-large', transform = ax.transAxes)

ax = subplot(4,2,8, projection = 'frameaxes')
            
bar( arange(30, 100*total_num_exp, 100), test_mean_spikes[0], width = 15, yerr = test_std_spikes[0], ecolor = 'k', color = '0.9' )
bar( arange(55, 100*total_num_exp, 100), test_mean_spikes[1], width = 15, yerr = test_std_spikes[1], ecolor = 'k', color = 'k' )


yticks(arange(0,6.5,2), [ '%d' % (x,) for x in arange(0,6.5,2) ])
ylim(0,8.1)


xticks(arange(50, 100*total_num_exp, 100), [ '%d' % (x+1,) for x in range(total_num_exp) ])

text(0.5, 0.78, 'after learning', fontsize = 'smaller', horizontalalignment = 'left', 
         verticalalignment = 'bottom', transform = ax.transAxes)

text(-0.24, 0.5, 'num. of spikes', fontsize = 'smaller', horizontalalignment = 'center', 
         verticalalignment = 'center', rotation = 90, transform = ax.transAxes)
text(-0.16, 0.5, 'per trial', fontsize = 'smaller', horizontalalignment = 'center', 
         verticalalignment = 'center', rotation = 90, transform = ax.transAxes)


xlabel('experiment \#')

text(-0.29,1.08,'H',fontsize = 'x-large', transform = ax.transAxes)


savefig("./directPattern.eps")
