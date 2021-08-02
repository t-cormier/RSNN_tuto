#!/usr/bin/python

from pypcsimplus.clusterUtils import *
import os.path
from datetime import datetime

machines = ['cluster1',  'cluster2',  'cluster3',
            'cluster4', 'cluster5', 'cluster6' ]

exp_group_dir = "compsim3_" + datetime.today().strftime("%Y%m%d_%H%M%S") 


if not os.path.exists(exp_group_dir):
    os.mkdir(exp_group_dir)


for i in range(6):
    spawn_remote_process(machines[i], 'python run_table_exp.py %d %s' % (i,exp_group_dir), os.path.join(exp_group_dir, "sim%d.out" % (i,)))

    