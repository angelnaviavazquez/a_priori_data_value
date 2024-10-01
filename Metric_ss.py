# -*- coding: utf-8 -*-
'''

Class that defines the metric to be used in the A priori Shapley estimation

@author:  Angel Navia Vázquez
Oct. 2024

'''

import numpy as np

class Metric():
    
    def __init__(self):
        self.name = 'ss'

    def get_stats(self, X, y):
        '''
        Computes the statistics from X, y data

        Parameters
        ----------
        X: array
            input data matrix

        y: array
            targets vector

        Returns
        -------
        stats : dict of statistical measures
        '''

        X = X.astype(float)
        # We map targets to (-1, 1)
        y = np.array(y).astype(float).reshape(-1, 1) * 2 - 1

        N = X.shape[0]
        stats = {'N':N}
        sumx = np.sum(X, axis=0)
        stats.update({'sumx': sumx})
        sumx2 = np.sum(X**2, axis=0)
        stats.update({'sumx2': sumx2}) 
        sumy = np.sum(y)
        stats.update({'sumy': sumy})
        sumy2 = np.sum(y**2)
        stats.update({'sumy2': sumy2}) 
        return stats

    def sim_cosine(self, v1, v2):
        '''
        Computes the cosine similarity between two vectors v1 and v2

        Parameters
        ----------
        v1: array
            vector 1

        v2: array
            vector 2

        Returns
        -------
        similarity: float value in (0,1)
        '''
        d = np.dot(v1.reshape(-1, 1).T, v2.reshape(-1, 1))
        v1Tv1 = np.sqrt(np.dot(v1.reshape(-1, 1).T, v1.reshape(-1, 1)))
        if v1Tv1 > 0:
            d = d / v1Tv1
        v2Tv2 = np.sqrt(np.dot(v2.reshape(-1, 1).T, v2.reshape(-1, 1)))
        if v2Tv2 > 0:
            d = d / v2Tv2
        return d.ravel()[0]

    def combine_statistics(self, stats_list):
        """
        Computes a single statistics dictionary from a list of statistics from different workers

        Parameters
        ----------
        stats_list: list of dicts
            list of data statistics

        Returns
        -------
        stats : dict
        """
        try:

            # We receive a single stats dict not in a list
            stats_list[0]
            if len(stats_list) == 1:
                return stats_list[0]
            else:
                stats = {}
                keys = stats_list[0].keys()
                for key in keys:
                    stats.update({key: 0})
                
                for stats_ in stats_list:
                    for key in keys:
                        stats[key] += stats_[key]

                return stats                
        except:
            return stats_list


    def get_vector(self, stats):  
        """
        Computes a vector from the stats, to also be used in external dot products
    
        Parameters
        ----------
        stats: dict
            data statistics

        Returns
        -------
        vector : array
        """
        N = stats['N']
        sumx = stats['sumx']
        NI = len(sumx)
        sumx2 = stats['sumx2']
        sumy = stats['sumy']
        sumy2 = stats['sumy2']

        mx = sumx / N
        my = sumy / N

        vary = sumy2/N - 2/N*my + my**2
        stdy = np.sqrt(np.abs(vary))

        varx = sumx2/N - 2/N*mx + mx**2
        stdx = np.sqrt(np.abs(varx))

        v = np.vstack( (mx.reshape((-1, 1)), stdx.reshape((-1, 1))) )
        my_ = np.ones((NI, 1)) * my
        v = np.vstack( (v, my) )
        stdy_ = np.ones((NI, 1)) * stdy
        v = np.vstack( (v, stdy) )

        return v

    def S(self, stats1, stats2):
        """
        Computes the Similarity between stats1 and stats2

        Parameters
        ----------
        stats1: dict
            statistics values 1

        stats2: dict
            statistics values 2

        Returns
        -------
        Similarity : float

        """
        s1 = self.combine_statistics(stats1)
        s2 = self.combine_statistics(stats2)

        v1 = self.get_vector(s1)
        v2 = self.get_vector(s2)

        #sim = 1 / np.mean((v1 - v2)**2)
        d2 = np.mean((v1 - v2)**2)
        sim = np.exp(-d2)
        
        return sim

