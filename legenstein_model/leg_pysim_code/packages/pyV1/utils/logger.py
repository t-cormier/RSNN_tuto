import logging
from mpi4py import MPI
import sys
#logging.basicConfig(format='%(name)s [%(asctime)s] -- %(module)s(%(lineno)d):  %(message)s', stream=sys.stdout)
logging.basicConfig(format='%(name)s [%(asctime)s] :  %(message)s', stream=sys.stdout)
logger=logging.getLogger('R%02d' % MPI.WORLD.rank)
logger.setLevel(logging.INFO)
