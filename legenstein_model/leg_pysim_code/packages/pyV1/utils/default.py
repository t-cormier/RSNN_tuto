import os

dirs = {}
home = os.environ['HOME']

if os.environ['USER']=='malte':
    dirs['ephys'] = home + '/work/data/ephys/'
    dirs['movies'] = home + '/work/data/movies/'
    dirs['pysim'] = home + '/saves/pysim/'


if os.environ['USER']=='schuch':
    dirs['ephys'] = home + '/igi/data/ephys/'
    dirs['movies'] = home + '/igi/data/movies/'
    dirs['pysim'] = home + '/igi/saves/pysim/'

