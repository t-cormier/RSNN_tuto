#!/usr/bin/python

from pypcsimplus.clusterUtils import *
import os.path
from datetime import datetime

machines = ['cluster1',  'cluster2',  'cluster3',
            'cluster4', 'cluster5', 'cluster6']


exp_group_dir = "pattern_final_" + datetime.today().strftime("%Y%m%d_%H%M%S") 

if not os.path.exists(exp_group_dir):
    os.mkdir(exp_group_dir)

i = 0
for m in machines:
    spawn_remote_process(m, 'python PatternRewardSTDPExperiment.py %d %s' % (i,exp_group_dir), os.path.join(exp_group_dir, "sim%d.out" % (i,)))
    i += 1
    