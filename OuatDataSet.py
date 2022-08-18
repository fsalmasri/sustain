import os.path as osp
from os.path import isfile, join
from os import listdir
from tqdm import tqdm
import numpy as np

import torch
from torch_geometric.data import Dataset, download_url, Data


class OuatDataSet(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        """
        :param root: dataset location, the folder is plit into raw (downloaded data) and processed (processed data)
        :param transform:
        :param pre_transform:
        :param pre_filter:
        """
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ['Bioreactor', 'Mixer', 'Connexion']

    @property
    def processed_file_names(self):
        return 'sustain.dataset'

    def download(self):
        # path = download_url(url, self.raw_dir)
        pass

    def process(self):
        for idx, p in enumerate(self.raw_paths):
            gFiles = listdir(p)
            for g_idx, gfile in enumerate(tqdm(gFiles, total=len(gFiles))):
                graph = np.load(join(p, gfile))
                node_feats, edge_index = self.get_attributes(graph)

                edge_index = torch.LongTensor(edge_index)
                node_feats = torch.LongTensor(node_feats)
                y = torch.FloatTensor(idx)

                data = Data(x=node_feats, edge_index=edge_index, y=y, graph=graph)
                torch.save(data, osp.join(self.processed_dir, f'data_{idx}_{g_idx}.pt'))


    def len(self):
        pass
        # return len(self.processed_file_names)

    def get(self, idx):
        pass
        # data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        # return data

    def get_attributes(self, g):

        dict_nodes ={}
        edge_index = []
        counter = 0
        for edge in g:
            for node in edge:
                if not node in dict_nodes:
                    dict_nodes[node] = counter
                    counter += 1

            edge_index.append([dict_nodes[edge[0]], dict_nodes[edge[1]]])

        node_feats = self.nodes_names_remapping(dict_nodes.keys())

        return node_feats, np.array(edge_index).T


    def nodes_names_remapping(self, g):

        nodes_embed = {
                "pumpHead": 0,
                "asepticDisconnector": 1,
                "mechanicDisconnector": 2,
                "quickCoupler": 3,
                "triclampConnector": 4,
                "bioreactorBag": 5,
                "twoDimensionalBag": 6,
                "threeDimensionalBag": 7,
                "mixerBag": 8,
                "bottle": 9,
                "plug": 10,
                "sensor": 11,
                "couplerReducer": 12,
                "straightFitting": 13,
                "lConnector": 14,
                "tConnector": 15,
                "yConnector": 16,
                "xConnector": 17,
                "asepticConnector": 18,
                "sipConnector": 19,
                "pinchClamp": 20,
                "hydrophobicFilter": 21,
                "hydrophilicFilter": 22,
                "tubing": 23
            }

        node_feats = []
        for node in g:
            if 'B' in node:
                node_feats.append(nodes_embed['bioreactorBag'])
            elif 'H' in node:
                node_feats.append(nodes_embed['hydrophobicFilter'])
            elif 'F' in node:
                node_feats.append(nodes_embed['hydrophilicFilter'])
            elif 'Q' in node:
                node_feats.append(nodes_embed['quickCoupler'])
            elif 'Y' in node:
                node_feats.append(nodes_embed['yConnector'])
            elif 'X' in node:
                node_feats.append(nodes_embed['xConnector'])
            elif 'P' in node:
                node_feats.append(nodes_embed['plug'])
            elif 'A' in node:
                node_feats.append(nodes_embed['asepticConnector'])
            if 'D' in node:
                node_feats.append(nodes_embed['asepticDisconnector'])
            elif 'S' in node:
                node_feats.append(nodes_embed['straightFitting'])
            elif 'M' in node:
                node_feats.append(nodes_embed['mixerBag'])
            elif 'R' in node:
                node_feats.append(nodes_embed['couplerReducer'])
            elif 'T' in node:
                node_feats.append(nodes_embed['triclampConnector'])

        return node_feats

dataset = OuatDataSet('data/')