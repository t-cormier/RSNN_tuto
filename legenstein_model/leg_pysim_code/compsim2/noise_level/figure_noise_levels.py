import os, sys
sys.path.append('../../packages')    
from tables import *
from pylab import *
from numpy import *
from math import *
from pypcsimplus.common import *
from pypcsimplus.Recordings import *
from pypcsimplus.Parameters import *
from frame import FrameAxes
import re

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Computer Modern Sans serif']})
rc('text', usetex=True)


if len(sys.argv) > 1:
    directory = sys.argv[1]
else:
    directory = last_created_dir('biofeed.*')

p = re.compile('biofeed.*\.h5')
entries = os.listdir(directory)
files = [ x for x in entries if p.match(x) ]    
files.sort()
print files
plot_colors = [ 'b', 'r', 'g', 'm', 'k']
col_n = 0
strong_syn_avg = []
weak_syn_avg = [] 
last_angle = []
noise_levels = []
for fname in files:
    h5file = openFile(os.path.join(directory,fname), mode = "r")

    all_p = constructParametersFromH5File(h5file)
    all_r = constructRecordingsFromH5File(h5file)
    
    h5file.close();
    
    p = all_p.biofeed
    ep = all_p.experiment
    
    r = all_r.biofeed
    
    noise_levels.append(p.OUScale)
    
    strong_syn_avg.append(average(r.weights[:p.numStrongTargetSynapses], 0))    
    weak_syn_avg.append(average(r.weights[p.numStrongTargetSynapses:], 0))
    
    target_w = hstack((ones(p.numStrongTargetSynapses)*p.Wmax, zeros(p.numWeakTargetSynapses)*p.Wmax))
    norm_target_w = target_w / sqrt(inner(target_w , target_w))
    normed_weights = r.weights.copy()
    for i in range(normed_weights.shape[1]):
        normed_weights[:,i] /= sqrt(inner(normed_weights[:,i], normed_weights[:,i]))    
    angle = arccos(dot(norm_target_w, normed_weights))
    
    last_angle.append(angle[-1])
    
all_together = zip(noise_levels,last_angle, strong_syn_avg, weak_syn_avg)

all_together.sort(lambda x,y: int(sign(x[0] - y[0])))

last_angle = [ x[1] for x in all_together ]
strong_syn_avg = [ x[2][-1] for x in all_together ]
weak_syn_avg = [ x[3][-1] for x in all_together ]
noise_levels = [ x[0] for x in all_together ]

figure(1)
ax = subplot(1,1,1, projection = 'frameaxes')    

bar(range(len(last_angle)), last_angle, width = 0.3, color = 'b')
xticks(arange(0.2,12,1), [ '%.1f' %(x,) for x in noise_levels])
xlim(-0.2,8.5)
xlabel('noise level')
ylabel('angular error [rad]')
yticks(arange(0.0,0.8,0.1), [ '%.1f' % (x,) for x in arange(0.0,0.8,0.1) ])
ylim(0,0.71)

savefig('noise_level.eps')

