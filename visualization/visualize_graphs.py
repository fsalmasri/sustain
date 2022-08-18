from os import listdir
from os.path import isfile, join
import numpy as np
import random
from visualization.visUtils import plotter_netx
import matplotlib.pyplot as plt

pathsLst = ['../data/raw/Bioreactor.v2', '../data/raw/Mixer', '../data/raw/Connexion.v2']


def random_visualizer(cls=0):

    Fpath = pathsLst[cls]
    gfiles = listdir(Fpath)
    random.shuffle(gfiles)

    # fig = plt.gcf()
    for c, graph in enumerate(gfiles):
        edges = np.load(join(Fpath, graph))

        plotter_netx(edges)
        c+=1
        plt.savefig(f'imgs.v2/{cls}/{graph[:-4]}.png')
        plt.clf()

        if c == 20:
            break


def visualize_graph(cls=0, g_id=0):


    Fpath = pathsLst[cls]
    edges = np.load(join(Fpath, f'{g_id}.npy'))
    fig, _ = plotter_netx(edges)
    # plt.savefig(f'imgs/2/{c}.png')
    # plt.clf()
    plt.show()


random_visualizer(2)
# visualize_graph(2, 'm_442353')