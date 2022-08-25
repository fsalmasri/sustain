from os import listdir
from os.path import isfile, join
import numpy as np
import random
from visualization.visUtils import plotter_netx
import matplotlib.pyplot as plt

pathsLst = ['../data/raw/Bioreactor.v2', '../data/raw/Mixer', '../data/raw/Connexion.v2', '../data/raw/Sanofi']


def random_visualizer(cls=0, n_imgs=10):
    '''
    :param cls: Class ID for the desired generated graphs
    :param c: Number of desired images to be saved
    :return:
    '''

    dst_dir = f'imgs.v2/{cls}'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    Fpath = pathsLst[cls]
    gfiles = listdir(Fpath)
    random.shuffle(gfiles)

    for c, graph in enumerate(gfiles):
        edges = np.load(join(Fpath, graph))

        plotter_netx(edges)
        plt.savefig(f'imgs.v2/{cls}/{graph[:-4]}.png')
        plt.clf()

        if c == n_imgs:
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