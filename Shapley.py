# -*- coding: utf-8 -*-
'''
Class that estimates the Shapley values given the list of permutations and their utilities

@author:  Angel Navia Vázquez
Oct. 2024

'''

import numpy as np
from itertools import chain, combinations
from math import factorial
import math

class Shapley():
    
    def __init__(self, V0):
        self.V0 = V0

    def powerset(self, iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def compute_scores(self, which_workers, models, verbose=True):
        """
        Computes the phi scores

        Parameters
        ----------
        models: list of tuples
            workers combinations and their utilities 

        Returns
        -------
        phis : the Shapley scores

        """
        NW = len(which_workers)
        phis = np.zeros(NW)
        # Average utilities
        AUs = np.zeros(NW)

        self.contribs = []

        for kselected in range(NW):
            selected = [which_workers[kselected]]
            remaining = list(set(which_workers) - set(selected))

            if verbose: 
                print('=' * 50)
            # emptyset
            w_ = np.copy(selected)
            key = [str(w) for w in w_]
            key = '_'.join(key)
            aux = models[key][1]
            if math.isnan(aux):
                aux = 0

            contrib_base = (aux - self.V0) 
            au_base = aux 

            phi0 = factorial(NW - 1) * contrib_base
            au0 = factorial(NW - 1) * au_base

            acum_fi = [phi0]
            acum_au = [au0]

            if verbose: 
                print(selected, [None], phi0, au0)
            self.contribs.append([selected, [None], phi0, contrib_base, au0, au_base])

            powerset_remaining = list(self.powerset(remaining))[1:]
            L = len(powerset_remaining)
            for k in range(L):
                active_workers = list(powerset_remaining[k])
                w_ = np.copy(active_workers)
                w_.sort()
                key = [str(w) for w in w_]
                key = '_'.join(key)
                M_active_workers = models[key][1]

                active_workers_plus_selected = selected + active_workers
                w_ = np.copy(active_workers_plus_selected)
                w_.sort()
                key = [str(w) for w in w_]
                key = '_'.join(key)
                M_active_workers_plus_selected = models[key][1]

                if math.isnan(M_active_workers):
                    M_active_workers = 0

                if math.isnan(M_active_workers_plus_selected):
                    M_active_workers_plus_selected = 0

                # Contribution of selected in this subset:
                N_active = len(active_workers)        
                
                contrib_base = M_active_workers_plus_selected - M_active_workers
                au_base = M_active_workers_plus_selected 

                contrib = contrib_base * factorial(N_active) * factorial(NW - 1 - N_active)
                au = au_base * factorial(N_active) * factorial(NW - 1 - N_active)
                
                if verbose:         
                    print(active_workers_plus_selected, active_workers, contrib, contrib_base, au, au_base)

                acum_fi.append(contrib)
                acum_au.append(au)

                self.contribs.append([active_workers_plus_selected, active_workers, contrib, contrib_base])

            phis[kselected] = np.sum(acum_fi) / factorial(NW)
            AUs[kselected] = np.sum(acum_au) / factorial(NW)

        self.AUs = AUs
        
        return phis.ravel()

