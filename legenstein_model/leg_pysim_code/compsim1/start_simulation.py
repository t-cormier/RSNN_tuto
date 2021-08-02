#!/usr/bin/python

import sys
sys.path.append('../packages')

import mpi_tools


machine_list = {}
for i in range(1,19):
    machine_list['cluster' + str(i)] = 2

attr_list = {}
    
mpi_tools.cluster_mpi_execute('python' , ['ou_more_syn_uniformized.py','final'], 
                              machine_list, attr_list, stdout = 'sim.out')