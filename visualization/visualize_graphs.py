from os import listdir
from os.path import isfile, join
import numpy as np
import random
from visualization.visUtils import plotter_netx

def random_visualizer(cls=0):


    pathsLst = ['../data/raw/Bioreactor', '../data/raw/Mixer', '../data/raw/Connexion']

    Fpath = pathsLst[cls]
    gfiles = listdir(Fpath)
    random.shuffle(gfiles)

    for graph in gfiles:
        edges = np.load(join(Fpath, graph))

        print(edges)

        plotter_netx(edges)
        # exit()



random_visualizer(2)