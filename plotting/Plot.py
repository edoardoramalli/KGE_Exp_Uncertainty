import os
import json
import matplotlib.pyplot as plt
import torch
from pykeen.triples import TriplesFactory
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import numpy as np


scenario = 'Case5'
general_path = './Scenario/{}/Dataset/'.format(scenario)

# path = os.path.join(general_path, '..', 'results/results.json')
#
# with open(path, 'r') as f:
#     diz = json.load(f)

# # print(diz['losses'])
#
# stopper = diz['stopper']
#
# f_stopper = stopper['frequency']
#
# x_val = list(range(f_stopper, f_stopper * len(stopper['results']) + f_stopper, f_stopper))
# y_val = stopper['results']
#
# x_loss = list(range(0, len(diz['losses'])))
# y_loss = diz['losses']
#
#
# plt.plot(x_loss, y_loss, 'b')
# plt.show()
#
# plt.plot(x_val, y_val, 'r')
# plt.show()


def plot_embeddings(reduction_model, kge_model, training_factory, filtering_rules, annotation=False):

    interesting_ids = {}

    int_id_list = []

    for entity in training_factory.entity_to_id:
        for rule in filtering_rules:
            if entity.startswith(rule):
                c_id = training_factory.entity_to_id[entity]
                interesting_ids[c_id] = rule
                int_id_list.append(c_id)

    labels = [training_factory.entity_id_to_label[i] for i in interesting_ids]

    entity_emb = kge_model.entity_embeddings.forward(indices=torch.tensor(int_id_list))

    if reduction_model == 'tsne':
        tsne = TSNE(random_state=1, n_iter=10000, metric="cosine")
        embs = tsne.fit_transform(entity_emb.detach().numpy())

    elif reduction_model == 'pca':
        pca = PCA(n_components=2)
        embs = pca.fit_transform(entity_emb.detach().numpy())

    else:
        raise ValueError('Model not supported')

    for i, emb in enumerate(embs):
        plt.plot(emb[0], emb[1], '{}.'.format(rules[interesting_ids[int_id_list[i]]]))
        if annotation:
            plt.annotate(labels[i], (embs[i][0], embs[i][1]))


    plt.show()


model = torch.load(os.path.join(general_path, '..', 'results', 'trained_model.pkl'), map_location=torch.device('cpu'))

training = TriplesFactory.from_path_binary(os.path.join(general_path, 'training'))

rules = {'TY_': 'r', 'R_': 'g', 'Y_': 'b', 'Author': 'm', 'T_': 'c', 'S_': 'k', 'U_': 'w', 'ID:': 'y'}


plot_embeddings('pca', model, training, rules, annotation=False)