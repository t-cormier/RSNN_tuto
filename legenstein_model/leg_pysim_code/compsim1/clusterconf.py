"""
configuration file for cluster
"""

controller = {'host' : 'figicl01',
              'engine_port' : 32100,
              'rc_port' : 32101,
              'task_port' : 32102}  

numEngPerNode = 1
clusterType = 'normal'

#
# Number of mpi engines. Used only if clusterType == mpi.
#
NumMpiEngines=30

#
# Lists the machine ip-s where engines will be started. Used only if clusterType == normal.
#
engines = {'cluster1' : numEngPerNode,
	   'cluster2' : numEngPerNode,
           'cluster3' : numEngPerNode,
           'cluster4' : numEngPerNode,
           'cluster5' : numEngPerNode,
           'cluster6' : numEngPerNode,
           'cluster7' : numEngPerNode,
	   'cluster8' : numEngPerNode,
           'cluster9' : numEngPerNode,
           'cluster10' : numEngPerNode,
           'cluster11' : numEngPerNode, 
           'cluster12' : numEngPerNode,
           'cluster13' : numEngPerNode,
           'cluster14' : numEngPerNode,
           'cluster15' : numEngPerNode,
           'cluster16' : numEngPerNode,
	   'cluster17' : numEngPerNode,
           'cluster18' : numEngPerNode,
           'cluster19' : numEngPerNode,
           'cluster20' : numEngPerNode,
           'cluster21' : numEngPerNode,
           'cluster22' : numEngPerNode,
           'cluster23' : numEngPerNode,
           'cluster24' : numEngPerNode,
           'cluster25' : numEngPerNode,
           'cluster26' : numEngPerNode,
           'cluster27' : numEngPerNode,
           'cluster28' : numEngPerNode,
           'cluster29' : numEngPerNode,
           'cluster30' : numEngPerNode }

import os
sshx = os.environ['RMSTDP_HOME'] + '/packages/sshx'

