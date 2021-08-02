import os, sys
sys.path.append('../packages')
from pylab import *
from tables import *
from numpy import *
from math import *
from pypcsimplus import *
import pypcsimplus
import pypcsimplus.common
from pypcsimplus.clusterUtils import *
from frame import FrameAxes
import numpy
import time

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('text', usetex=True)

random.seed(123525)

if len(sys.argv) > 1:
    h5filename = sys.argv[1]

else:
    h5filename = last_file(".*\.h5")
    
print " loading h5 filename : ", h5filename

h5file = openFile(h5filename, mode = "r", title = "Biofeedback DASTDP Experiment results")

p = constructParametersFromH5File(h5file)
r = constructRecordingsFromH5File(h5file)


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
    
analysis_intervals = [ (300,360), (600,660), (900,960), (1140,1200) ]

gray_colors = ['0.75', '0.6', '0.4', '0.0']
plot_colors = [ 'r', 'g', 'b', 'm' ]     

IntervalLength = analysis_intervals[0][1] - analysis_intervals[0][0]
    
w_hist_array = []
numBins = 30
for pl_i in range(len(analysis_intervals)):    
    ind_start = analysis_intervals[pl_i][0] / (p.Tsim / len(other_weights[0]))
    ind_end = analysis_intervals[pl_i][1] / (p.Tsim / len(other_weights[0]))                
    w_hist_array.append(histogram(mean(other_weights[:,ind_start:ind_end]/p.Wmax,1), bins=numBins, range = (0,1), normed = False)[0]/float(len(other_weights)))
                

avg_rates = []
numBins = 20        
for t in range(len(analysis_intervals)):
    avg_rates.append([])
    for i in range(len(r.spikes[:p.nExcNeurons])):
        avg_rates[t].append( calc_rate_2(clip_window(r.spikes[i], analysis_intervals[t][0], analysis_intervals[t][1]), 1, 1, IntervalLength) )

rate_hist_array = []
for t in range(len(analysis_intervals)):
    rate_hist_array.append(histogram(avg_rates[t], bins = numBins, range = (0,14), normed = False)[0]/float(len(avg_rates[t])))
    
#=============================================================================================    

f = figure(1, figsize = (8,6), facecolor = 'w')        
f.subplots_adjust(top= 0.95, left = 0.085, bottom = 0.11, right = 0.95, hspace = 0.36, wspace = 0.30)

clf()

plot_labels = ['300-360 sec', '600-660 sec', '900-960 sec', '1140-1200 sec' ]

ax = subplot(2,2,1, projection = 'frameaxes')
for pl_i in range(len(analysis_intervals)):
    plot(arange(0,1.0+0.9/len(w_hist_array[pl_i]),(1.0+1.0/len(w_hist_array[pl_i]))/len(w_hist_array[pl_i])), w_hist_array[pl_i], linewidth = 2, color = plot_colors[pl_i], label = plot_labels[pl_i] )

legend(prop = matplotlib.font_manager.FontProperties(size = 'xx-small'))
    
xlabel('synaptic weight $(w/w_{max})$')
ylabel('frac. synapses [\%]')    
yticks(arange(0,0.41,0.10), [ "%d" % (x) for x in arange(0, 0.41, 0.10)*100 ] )
xticks(arange(0,1.01,0.2), [ "%.1f" % (x) for x in arange(0,1.01,0.2) ] )
ylim(0,0.40)

text(-0.2,1.04, 'A',fontsize = 'x-large', transform = ax.transAxes)

plot_labels = ['300-360 sec', '600-660 sec', '900-960 sec', '1140-1200 sec' ]

ax = subplot(2,2,2, projection = 'frameaxes')
for pl_i in range(len(analysis_intervals)):
    plot(arange(0,16.0+15.9/len(rate_hist_array[pl_i]),(16.0+15.9/len(rate_hist_array[pl_i]))/len(rate_hist_array[pl_i])), rate_hist_array[pl_i], linewidth = 2, color = plot_colors[pl_i], label = plot_labels[pl_i] )
    
ylabel('frac. neurons [\%]')    
xlabel('firing rate [Hz]')
xticks(arange(0,16.1,4), [ "%d" % (x) for x in arange(0,16.1,4) ])    
yticks(arange(0, 0.201, 0.05), [ "%d" % (x) for x in arange(0, 0.201, 0.05)*100 ])
ylim(0, 0.201)

text(-0.2, 1.04, 'B', fontsize = 'x-large', transform = ax.transAxes)
    
legend(prop = matplotlib.font_manager.FontProperties(size = 'xx-small'))

IPcontroller = {'host' : 'cluster1', 'engine_port' : 32100, 'rc_port' : 32101, 'task_port' : 32102}
IPcluster = Cluster(ClusterConfig(configFile='./clusterconf.py', controller=IPcontroller))
IPcluster.start(waitafter = 3.0)
IPcluster.connect()
rc = IPcluster.getRemoteControllerClient()


print "len(rc) is ", len(rc)
while len(rc) < 30:
    print "len(rc) is ", len(rc)
    time.sleep(1)

rc.execute('import pypcsimplus')
rc.execute('from pypcsimplus import *')
rc.execute('import pypcsimplus.common')    


corrDT = 0.2e-3
leftCorrBound = -100e-3
rightCorrBound = 100e-3
numNrnPairs = 200
pairs = set([])
while len(pairs) < numNrnPairs:
    rnd_pair  = (random.randint(0,p.nExcNeurons-1),random.randint(0,p.nExcNeurons-1))
    if not rnd_pair[0] == rnd_pair[1]:
        pairs.add(rnd_pair)

i = 0
figure_letter = ['C', 'D']
for pl_i in (0,len(analysis_intervals)-1):
    print "processing cross_covariance for interval %d" % (pl_i)
    leftSimT = analysis_intervals[pl_i][0]
    rightSimT = analysis_intervals[pl_i][1]
    
    arg_list = []
    for n1, n2 in pairs:        
        arg_list.append([r.spikes[n1], r.spikes[n2], corrDT, (leftSimT, rightSimT), (leftCorrBound, rightCorrBound)])
                
    corr = rc.map(pypcsimplus.common.cross_covariance_spikes_1arg, arg_list)
    
    avg_corr = mean(vstack(tuple(corr)), 0)
    
    ax = subplot(2,2,i + 3, projection = 'frameaxes')     
    text(-0.2,1.04, figure_letter[i],fontsize = 'x-large', transform = ax.transAxes)
    vlines( arange(int(leftCorrBound/corrDT) * corrDT, int(rightCorrBound/corrDT) * corrDT + corrDT/100, corrDT), zeros(len(avg_corr)), avg_corr, linewidth = 1)
    xlim(leftCorrBound,rightCorrBound)
    xticks(arange(-100e-3,101e-3,50e-3), [  '%d' % (x) for x in arange(-100,101,50)])
    
    xlabel("$\\tau$ [ms]")
    ylabel("$C(\\tau)\quad (\\cdot 10^{-4})$")
    yticks(arange(-0.0005,0.00101,0.0005), [ "%d" % (x) for x in arange(-5,11,5)])
    
    ylim(-0.0005,0.00101)
    i += 1

    
IPcluster.stop()
    
savefig('correlations_weight_depend.eps')
