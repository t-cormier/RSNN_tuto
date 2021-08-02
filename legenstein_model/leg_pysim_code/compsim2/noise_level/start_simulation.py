#!/usr/bin/python

from pypcsimplus.clusterUtils import *
import os.path
from datetime import datetime
import sys

machines = ['cluster%d' % (i,) for i in range(1,30)]

runName = 'default'

exp_group_dir = "biofeed_noise_" + runName + "_" + datetime.today().strftime("%Y%m%d_%H%M%S")  

noiseLevels = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4, 2.2, 3.6 ]

if not os.path.exists(exp_group_dir):
    os.mkdir(exp_group_dir)

for i in range(len(noiseLevels)):
    spawn_remote_process(machines[i], 'python BiofeedExperiment.py %s %.1f %s' % (runName, noiseLevels[i],exp_group_dir), os.path.join(exp_group_dir, "sim%d.out" % (i,)))

    