from os import listdir
from os.path import isfile, join
import numpy as np
import random
from visualization.visUtils import plotter_netx

def random_visualizer(cls=0):


    pathsLst = ['dataset/OUAT_DS/Bioreactor', 'dataset/OUAT_DS/Mixer', 'dataset/OUAT_DS/Connexion2']

    Fpath = pathsLst[cls]
    gfiles = listdir(Fpath)
    random.shuffle(gfiles)

    for graph in gfiles:
        edges = np.load(join(Fpath, graph))

        print(edges)

        plotter_netx(edges)
        # exit()



random_visualizer(2)