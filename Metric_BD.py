# -*- coding: utf-8 -*-
'''

Class that defines the metric to be used in the A priori Shapley estimation

@author:  Jesús Cid Sueiro
Oct. 2024

'''

import numpy as np


class Metric():

    def __init__(self, feature_map='rx1y', mapping='BD'):
        """
        Parameters
        ----------
        feature_map: str, optional (default='standard')
            Type of feature mapping: 'rxy', 'rx1y'
        mapping: None or str, optional (defaul='linear')
            Type of mapping from the statistics into a [-1, 1] interval.
            Available options are:
            'scalar':
            'scalarbayes':
            'multidim':
        """

        self.feature_map = feature_map
        self.mapping = mapping
        self.name = f'BD_{self.feature_map}_{self.mapping}'

    def get_features(self, X, y):
        """
        Compute the feature matrix for the computation of statistics:
        """

        X = X.astype(float)
        # We map targets to (-1, 1)
        y = np.array(y).astype(float).reshape(-1, 1) * 2 - 1

        # Only rxy features available. Add other cases if necessary
        if self.feature_map == 'rxy':
            Z = X * y

        elif self.feature_map == 'rx1y':
            # Add an all ones colum to CX
            ones = np.ones((X.shape[0], 1))
            Z = np.hstack((X, ones)) * y

        return Z

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

        Z = self.get_features(X, y)

        if self.mapping in {'spheric'}:
            stats = {'N': Z.shape[0],
                     'sumz': np.sum(Z, axis=0),
                     'sum_mean_z2': np.sum(np.mean(Z**2, axis=1), axis=0)}
        if self.mapping in {'scalar', 'cosine', 'scalarbayes'}:
            stats = {'N': Z.shape[0],
                     'sumz': np.sum(Z, axis=0),
                     'sumz2': np.sum(Z**2, axis=0)}
        elif self.mapping == "multidim":
            stats = {'N': Z.shape[0],
                     'sumz': np.sum(Z, axis=0),
                     'sumz2': np.sum(Z**2, axis=0),
                     'Rz': Z.T @ Z}
        elif self.mapping == 'scalar01':
            stats = {'N0': np.sum(1 - y),
                     'N1': np.sum(y),
                     'sumz_0': -(1 - y) @ Z,
                     'sumz_1': y @ Z,
                     'sumz2_0': (1 - y) @ (Z**2),
                     'sumz2_1': y @ (Z**2)}

        return stats

    def combine_statistics(self, stats_list):
        """
        Computes a single statistics dictionary from a list of statistics from
        different workers

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
                stats = {key: 0 for key in stats_list[0]}

                for stats_ in stats_list:
                    for key in stats:
                        stats[key] += stats_[key]

                return stats
        except:
            return stats_list

    def S(self, stats_A, stats_B):
        """
        Computes the Similarity between stats_A and a reference stats_B

        Parameters
        ----------
        stats_A: dict
            statistics of population A

        stats_B: dict
            statistics of population B

        Returns
        -------
        Similarity : float
        """

        eps = 1e-100

        sA = self.combine_statistics(stats_A)
        sB = self.combine_statistics(stats_B)

        if self.mapping == 'scalarbayes':
            # Compute Bayesian estimates of means and variances based on the
            # Normal-inverse-gamma prior (formulae taken from Sec. 6 in
            # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)

            # Below, I use k_n = 1/Vn instead of variable Vn in
            # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
            # to avoid missleading with variance estimate vA
            # k_0A can be interpreted as the "equivalent sample size" of the
            # prior on the variance

            # Hyperparameters
            k_0A = 10
            alpha_0A = 1
            beta_0A = 1
            # Hyperparameters
            k_0B = 10
            alpha_0B = 1
            beta_0B = 1

            # Dataset sizes
            nA = sA['N']
            nB = sB['N']

            # Sample mean of each feature in each dataset
            # mA = sA['sumz'] / nA
            # mB = sB['sumz'] / nB

            # Bayesian update
            k_nA = k_0A + nA
            m_nA = sA['sumz'] / k_nA
            alpha_nA = alpha_0A + nA / 2
            beta_nA = beta_0A + 0.5 * (sA['sumz2'] - m_nA**2 * k_nA)
            vA = beta_nA / (alpha_nA - 1) + eps

            # Bayesian update
            k_nB = k_0B + nB
            m_nB = sB['sumz'] / k_nB
            alpha_nB = alpha_0B + nB / 2
            beta_nB = beta_0B + 0.5 * (sB['sumz2'] - m_nB**2 * k_nB)
            vB = beta_nB / (alpha_nB - 1) + eps

            # This is the Battacharya coefficient.
            vAB = (vA + vB) / 2
            # logdetA = np.sum(np.log(vA + eps)) / 4
            # logdetB = np.sum(np.log(vB + eps)) / 4
            # logdetAB = np.sum(np.log(vAB + eps)) / 2
            # logsim = (logdetA + logdetB - logdetAB
            #           - np.sum((mA - mB)**2 / (8 * vAB)))
            # sim = np.exp(logsim)

            # WARNING: Next lines assume vA and vB 1D arrays or column vectors
            d = np.sum(m_nA * m_nB / vAB)
            # vAnorm = np.sqrt(np.sum(mA**2 / vAB))
            # vBnorm = np.sqrt(np.sum(mB**2 / vAB))
            normA = np.sqrt(np.sum(m_nA**2 / vA))
            normB = np.sqrt(np.sum(m_nB**2 / vB))
            sim = d / (normA * normB + eps)

        if self.mapping == 'scalar':
            # Dataset sizes
            nA = sA['N']
            nB = sB['N']

            # Sample mean of each feature in each dataset
            mA = sA['sumz'] / nA
            mB = sB['sumz'] / nB

            # # Variance of each feature in each dataset
            vA = sA['sumz2'] / nA - mA**2 + eps
            vB = sB['sumz2'] / nB - mB**2 + eps

            # This is the Battacharya coefficient.
            vAB = (vA + vB) / 2

            # WARNING: Next lines assume vA and vB 1D arrays or column vectors
            d = np.sum(mA * mB / vAB)
            # vAnorm = np.sqrt(np.sum(mA**2 / vAB))
            # vBnorm = np.sqrt(np.sum(mB**2 / vAB))
            normA = np.sqrt(np.sum(mA**2 / vA))
            normB = np.sqrt(np.sum(mB**2 / vB))
            sim = d / (normA * normB + eps)

        elif self.mapping == 'spheric':

            # This is equivalent to "scalar" but variances are averaged acoss
            # features

            # Dataset sizes
            nA = sA['N']
            nB = sB['N']

            # Sample mean of each feature in each dataset
            mA = sA['sumz'] / nA
            mB = sB['sumz'] / nB

            # # Variance of each feature in each dataset
            vA = sA['sum_mean_z2'] / nA - np.mean(mA**2) + eps
            vB = sB['sum_mean_z2'] / nB - np.mean(mB**2) + eps

            # This is the Battacharya coefficient.
            vAB = (vA + vB) / 2

            # WARNING: Next lines assume vA and vB 1D arrays or column vectors
            d = np.sum(mA * mB / vAB)
            # vAnorm = np.sqrt(np.sum(mA**2 / vAB))
            # vBnorm = np.sqrt(np.sum(mB**2 / vAB))
            normA = np.sqrt(np.sum(mA**2 / vA))
            normB = np.sqrt(np.sum(mB**2 / vB))
            sim = d / (normA * normB + eps)

        elif self.mapping == 'scalar01':

            eps = 1e-100
            sim = 0

            for i in ['0', '1']:
                # Dataset sizes
                nA = sA['N' + i]
                nB = sB['N' + i]

                if nA * nB == 0:
                    break

                # Sample mean of each feature in each dataset
                mA = sA['sumz_' + i] / nA
                mB = sB['sumz_' + i] / nB

                # # Variance of each feature in each dataset
                vA = sA['sumz2_' + i] / nA - mA**2
                vB = sB['sumz2_' + i] / nB - mB**2

                # # This is just to be a bit more precise:
                # vA = vA * nA / (nA - 1) + eps
                # vB = vB * nB / (nB - 1) + eps
                vA = vA + eps
                vB = vB + eps

                # This is the Battacharya coefficient.
                vAB = (vA + vB) / 2

                # WARNING: Next lines assume vA,vB 1D arrays or column vectors
                d = np.sum(mA * mB / vAB)
                vAnorm = np.sqrt(np.sum(mA**2 / vA)) + eps
                vBnorm = np.sqrt(np.sum(mB**2 / vB)) + eps
                sim += d / (vAnorm * vBnorm + eps)

            sim = sim / 2

        elif self.mapping == 'multidim':
            # Compute Bayesian estimates of means and variances based on the
            # Normal-inverse-gamma prior (formulae taken from Sec. 6 in
            # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)

            # Below, I use k_n = 1/Vn instead of variable Vn in
            # https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
            # to avoid missleading with variance estimate vA
            # k_0A can be interpreted as the "equivalent sample size" of the
            # prior on the variance

            eps = 1e-15

            # Dataset sizes
            nA = sA['N']
            nB = sB['N']

            # Sample mean of each feature in each dataset
            mA = sA['sumz'] / nA
            mB = sB['sumz'] / nB

            # Data dimension (no. of features)
            dim = len(mA)

            # Convert to column vectors
            mA = np.array([mA]).T
            mB = np.array([mB]).T

            # # Variance of each feature in each dataset
            vA = sA['Rz'] / nA - mA @ mA.T
            vB = sB['Rz'] / nB - mB @ mB.T

            # # This is just to be a bit more precise:
            # vA = vA * nA / (nA - 1) + eps * np.eye(dim)
            # vB = vB * nB / (nB - 1) + eps * np.eye(dim)
            vA = vA + eps * np.eye(dim)
            vB = vB + eps * np.eye(dim)

            # This is the Battacharya coefficient.
            vAB = (vA + vB) / 2

            # WARNING: Next lines assume vA and vB 1D arrays or column vectors
            d = (mA.T @ np.linalg.inv(vAB) @ mB)[0, 0]
            vAnorm = np.sqrt(mA.T @ np.linalg.inv(vA) @ mA)[0, 0]
            vBnorm = np.sqrt(mB.T @ np.linalg.inv(vB) @ mB)[0, 0]
            sim = d / (vAnorm * vBnorm + eps)

        elif self.mapping == 'cosine':
            # WARNING: Next lines assume vA and vB 1D arrays or column vectors
            d = mA.T @ mB
            vAnorm = np.sqrt(mA.T @ mA)
            vBnorm = np.sqrt(mB.T @ mB)
            sim = d / (vAnorm * vBnorm + eps)

        if (sim**2 > 1) or np.isnan(sim):
            print("WARNING: similarities are above 1 or nan")
            breakpoint()

        # v1 = self.get_vector(s1)
        # v2 = self.get_vector(s2)
        # return self.sim_cosine(v1, v2).ravel()[0]
        return sim
