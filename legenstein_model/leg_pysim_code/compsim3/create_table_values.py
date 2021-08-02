import os, sys
    
from tables import *
from pylab import *
from numpy import *
from math import *
from pypcsimplus.common import *
import re

sys.path.append('..')

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('text', usetex=True)


import Constraints
reload(Constraints)

if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    directory = last_created_dir(".*")

h5filenames = os.listdir(directory)
patt = re.compile(".*\.h5")
h5filenames = [ os.path.join(directory, x) for x in h5filenames if patt.match(x) ]

h5filenames.sort()

params = []
strong_syn_avg = []
weak_syn_avg = []
est_strong_weight_change = []
est_weak_weight_change = []
actual_strong_weight_change = []
actual_weak_weight_change = []
i = 0
for filename in h5filenames:    
    h5file = openFile(filename, mode = "r", title = "Biofeedback DASTDP Experiment results")
    strong_syn_avg = h5file.getNode("/AverageSynapseWeights", "strong_avg").read()    
    weak_syn_avg = h5file.getNode("/AverageSynapseWeights", "weak_avg").read()    
    params.append( h5file.getNode("/","parameters")[0] )
    DW_over_DT_strong , DW_over_DT_weak, Tconv_strong, Tconv_weak, LeftSide_weak, RightSide_weak, LeftSide_strong, RightSide_strong = \
    Constraints.checkConstraints(params[i]["synTau"], params[i]["NumSyn"], params[i]["ratioStrong"], params[i]["WmaxTrue"],
           params[i]["inputRate"], params[i]["Rbase"], 
           params[i]["stdpAposTrue"], params[i]["stdpAnegTrue"], params[i]["stdpTaupos"], params[i]["stdpTauneg"], 
           params[i]["DAStdpRate"], params[i]["DATraceDelay"], params[i]["DATraceTau"], 
           params[i]["DATraceShape"], params[i]["rewardDelay"], 
           params[i]['KappaApos'], params[i]["KappaAneg"], params[i]["KappaTaupos"], params[i]["KappaTauneg"], 
           params[i]["KappaTe"], params[i]["numAdditionalTargetSynapses"], params[i]["KappaTaupos2" ],  
           params[i]["KappaTauneg2"], params[i]["KernelType"] )
    
    est_strong_weight_change.append( params[i]['Tsim'] / Tconv_strong  )
    est_weak_weight_change.append( params[i]['Tsim'] / - Tconv_weak )
    actual_strong_weight_change.append( (strong_syn_avg[-1] - strong_syn_avg[0]) / params[i]["Wmax"] * 2 )
    actual_weak_weight_change.append( (weak_syn_avg[-1] - weak_syn_avg[0]) / params[i]["Wmax"] * 2 )    
    h5file.close()
    i += 1



f = figure(1, figsize = (6,4), facecolor = 'w')
clf()

f.subplotpars.left = 0.18
f.subplotpars.right = 0.96
f.subplotpars.top = 0.86
f.subplotpars.bottom = 0.12
f.subplotpars.hspace = 0.55

x_est_bar = [20 + i * 100 for i in range(6)]
x_act_bar = [60 + i * 100 for i in range(6)]

rect_color = '0.92'

a = subplot(2,1,1)
a.set_frame_on(False)
rect = Rectangle((200,-0.65), 100, 1.65, fill = True, facecolor = rect_color, edgecolor = rect_color)
rect2 = Rectangle((400,-0.65), 100, 1.65, fill = True, facecolor = rect_color, edgecolor = rect_color)
a.add_patch(rect)
a.add_patch(rect2)
bar(x_est_bar + x_act_bar, est_strong_weight_change + actual_strong_weight_change , color  = ['k' for i in range(6)] + ['0.5' for i in range(6) ], width = 20)
xlim(0,600)
a.hlines(0,0,600)
yticks( arange(-0.5,1.01,0.5), [ "%.1f" % i for i in arange(-0.5, 1.01, 0.5)] )
ylim(-0.65,1)
a.vlines(0.2,-1,1)
a.yaxis.tick_left()
ylabel('$\\Delta w\\: (w^{*} = w_{max})$', fontsize = 'small')
setp( a.xaxis, visible = False )
for i in range(1,6):
    vlines(i*100, -1, 1, color = 'k', linestyle = 'dotted')
text(-0.21, 1.2,'A', fontsize = 'x-large', transform = a.transAxes)

a = subplot(2,1,2)
a.set_frame_on(False)
rect = Rectangle((100,-1.65), 100, 2.65, fill = True, facecolor = rect_color, edgecolor = rect_color)
rect2 = Rectangle((500,-1.65), 100, 2.65, fill = True, facecolor = rect_color, edgecolor = rect_color)
a.add_patch(rect)
a.add_patch(rect2)
bar(x_est_bar + x_act_bar, est_weak_weight_change + actual_weak_weight_change, color  = ['k' for i in range(6)] + ['0.5' for i in range(6) ], width = 20)

xlim(0,600)
a.hlines(0,0,600)
a.vlines(0.2,-1.2,1)
a.yaxis.tick_left()
setp( a.xaxis, visible = False )
yticks( arange(-1.0,0.6,0.5), [ "%.1f" % i for i in arange(-1.0, 0.6, 0.5)] )
ylim(-1.2, 0.65)
ylabel('$ \\Delta w \\, (w^{*} = 0) $', fontsize = 'small')
for i in range(1,6):
    vlines(i*100, -1.2, 1, color = 'k', linestyle = 'dotted')
for i in range(6):
    text(45 +  i * 100, -1.95, '%d' % (i+1,) , fontsize = 'large')
text(-110, -2.00, 'Exp. \#' , fontsize = 'large')
text(-0.21, 1.12,'B', fontsize = 'x-large', transform = a.transAxes)

savefig('./condition_bars.eps')
