# -*- coding: utf-8 -*-
'''

Class that computes the model utilities for all combinations among workers 

@author:  Angel Navia Vázquez
Oct. 2024

'''

import numpy as np
from itertools import chain, combinations
from math import factorial
from LC_model import Model as MLmodel
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

class Models():
    
    def __init__(self, Xtr_chunks, ytr_chunks, Xval=None, yval=None, worker_stats_dict=None, ref_stats=None, metric=None, brute_force_model='LR'):
        '''
        self.NW = len(self.Xtr_chunks)
        '''
        self.Xtr_chunks = Xtr_chunks
        self.ytr_chunks = ytr_chunks
        self.Xval = Xval
        self.NI = Xval.shape[1]
        self.yval = yval
        self.NW = len(Xtr_chunks)
        self.which_workers = list(range(self.NW))

        # Only for apriori
        self.worker_stats_dict = worker_stats_dict
        self.ref_stats = ref_stats
        self.metric = metric
        self.brute_force_model = brute_force_model

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def M_ACC(self, o, y):
        # The performance metric for a classification problem. The larger the better
        o = (o > 0.5).astype(float)
        acc = float(np.mean(o == y.astype(float)))
        return acc

    def get_models(self, utility):
        """
        Computes the models and utilities of all of the workers combinations

        Parameters
        ----------
        models: list of tuples
            workers combinations and their utilities

        Returns
        -------
        models : the models and utilities

        """

        self.models_dict = {}

        combination_workers = list(self.powerset(self.which_workers))[1:]
        Ncomb = len(combination_workers)

        for kcomb in range(Ncomb):
            workers = combination_workers[kcomb]
            selected = list(workers)

            if utility == 'ACC': # Shapley brute force
                # model = MLmodel(self.NI)
                # model.fit(self.Xtr_chunks, self.ytr_chunks, selected,
                #           self.Nmaxiter, self.conv_stop, self.mu,
                #           self.Nbatch, self.store_gradients, verbose=False)
                # Probamos con scikit learn
                Xtr = self.Xtr_chunks[selected[0]]
                ytr = self.ytr_chunks[selected[0]]

                for kworker in selected[1:]:
                    Xtr = np.vstack((Xtr, self.Xtr_chunks[kworker]))
                    ytr = np.hstack((ytr, self.ytr_chunks[kworker]))

                #print('=========== 1000 ====================')
                # When the dataset only has one class, the model cannot be trained
                which_class = list(set(ytr))
                if len(which_class) > 1:
                    if self.brute_force_model == 'LR':
                        model = LogisticRegression(fit_intercept=True, max_iter=1000)
                        model.fit(Xtr, ytr)
                    elif self.brute_force_model == 'NN':
                        NF = Xtr.shape[1]
                        mlp_size = (int(NF/2), int(NF/4))
                        model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=mlp_size, random_state=1)
                        model.fit(Xtr, ytr)
                    else:
                        print('ERROR at Models: Model not supported')
                        return

                    preds_val = model.predict_proba(self.Xval)[:, 1].ravel()

                    if utility == 'CE':
                        M_selected = self.M_CE(preds_val, self.yval)

                    if utility == 'AUC':
                        M_selected = self.M_AUC(preds_val, self.yval)

                    if utility == 'ACC':
                        M_selected = self.M_ACC(preds_val, self.yval)
                else:
                    model = None
                    M_selected = 0.5

            else:  # A priori Shapley
                train_stats = []

                for kworker in selected:
                    train_stats.append(self.worker_stats_dict[kworker])

                M_selected = self.metric.S(train_stats, self.ref_stats)
                model = train_stats

            w_ = np.copy(workers)
            w_.sort()
            key = [str(w) for w in w_]
            key = '_'.join(key)
            self.models_dict.update({key: [selected, M_selected, model]})

        return self.models_dict
