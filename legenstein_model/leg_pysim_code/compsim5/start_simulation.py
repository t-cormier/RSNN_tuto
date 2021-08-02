#!/usr/bin/python

import os
from subprocess import *
import time
import re

os.system('nohup python SpeechRewardSTDPExperiment.py --ipcontroller-host=cluster1 < /dev/null &> sim.out &')
print "Done."
