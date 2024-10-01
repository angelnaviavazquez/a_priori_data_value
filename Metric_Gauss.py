# -*- coding: utf-8 -*-
'''

Class that defines the metric to be used in the A priori Shapley estimation

@author:  Jesús Cid Sueiro
Oct. 2024

'''

import numpy as np
import scipy.stats as scpst


class Metric():

    def __init__(self, option='no_ones', mapping='scalar', ref='paired'):
        """
        This metric estimates the quality of the data by assuming bayesian
        classifier assuming Gaussian class-conditional distribution with equal
        variances.

        Parameters
        ----------
        feature_map: str, optional (default='standard')
            Type of feature mapping: 'no_ones', 'ones'

        mapping: str, optional (defaul='linear')
            Type of linear gaussian classifier.
            Available options are:
                'spheric': isotropic gaussians
                'scalar': diagonal variance matrices
                'multidim': arbitrary variance matrices
        ref: str, optional (defaul='paired')
            Type of evaluation of the error rate of the classifiers
            Available options are:
                'paired': Same type of variances than the classifier
                'hybrid': Arbitrary variance matrices
                'empiric': Error rates is empiricall estimated based on the
                reference data

            'multidim':
        """

        self.option = option
        self.mapping = mapping
        self.ref = ref
        self.name = f'Gauss_{self.mapping}'

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

        # Take columns with features only (the 1st column is expected to be
        # all-ones)
        if self.option == 'ones':
            Xf = X[:, 1:]
        elif self.option == 'no_ones':
            Xf = X
        else:
            exit(f"ERROR: Unknown option {self.option}")

        # Common components:
        stats = {'N0': np.sum(1 - y),
                 'N1': np.sum(y),
                 'sumx_0': (1 - y) @ Xf,
                 'sumx_1': y @ Xf}

        if self.mapping == 'spheric':
            stats.update({'sum_mean_x2_0': (1 - y) @ np.mean((Xf**2), axis=1),
                          'sum_mean_x2_1': y @ np.mean((Xf**2), axis=1)})

        elif self.mapping == 'scalar':
            stats.update({'sumx2_0': (1 - y) @ (Xf**2),
                          'sumx2_1': y @ (Xf**2)})

        elif self.mapping == "multidim":
            Zf_0 = ((1 - y) * Xf.T).T
            Zf_1 = (y * Xf.T).T
            stats.update({'Rx_0': Zf_0.T @ Zf_0,
                          'Rx_1': Zf_1.T @ Zf_1})

        if self.ref == 'hybrid':
            Zf_0 = ((1 - y) * Xf.T).T
            Zf_1 = (y * Xf.T).T
            stats.update({'Rx_0': Zf_0.T @ Zf_0,
                          'Rx_1': Zf_1.T @ Zf_1})

        elif self.ref == "empiric":
            stats.update({'y': np.array([y]).T,
                          'X': Xf})

        return stats

    def combine_statistics(self, stats_list, add_data=False):
        """
        Computes a single statistics dictionary from a list of statistics from
        different workers

        Parameters
        ----------
        stats_list: list of dicts
            list of data statistics
        add_data: bool, optional (default=False)
            If true data may be added to the output dictionry if requested in
            the input dictionaries

        Returns
        -------
        stats : dict
        """

        if add_data:
            no_keys = {}
        else:
            no_keys = {'X', 'y'}

        if type(stats_list) == list:

            # We receive a single stats dict not in a list
            if len(stats_list) == 1:
                # Return dict without data entries
                stats = {key: v for key, v in stats_list[0].items()
                         if key not in no_keys}

            else:
                stats = {key: 0 for key in stats_list[0]
                         if key not in no_keys}

                for stats_ in stats_list:
                    for key in stats:
                        if key in {'X', 'y'}:
                            if stats[key] is 0:
                                stats[key] = stats_[key]
                            else:
                                stats[key] = np.vstack(
                                    (stats[key], stats_[key]))
                        else:
                            stats[key] += stats_[key]
        else:
            stats = {key: v for key, v in stats_list.items()
                     if key not in no_keys}

        return stats

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

        sA = self.combine_statistics(stats_A)
        if self.ref == 'empiric':
            sB = self.combine_statistics(stats_B, add_data=True)
        else:
            sB = self.combine_statistics(stats_B)

        # Dataset sizes
        nA0, nA1 = sA['N0'], sA['N1']
        nB0, nB1 = sB['N0'], sB['N1']

        # Class prior probabilities
        PA0 = nA0 / (nA0 + nA1)
        PA1 = nA1 / (nA0 + nA1)
        PB0 = nB0 / (nB0 + nB1)
        PB1 = nB1 / (nB0 + nB1)

        # ##################
        # Compute classifier

        if self.mapping == 'spheric0':

            eps = 1e-100

            # Sample mean of each feature in each dataset
            if nA0 > 0:
                mA0 = sA['sumx_0'] / nA0
            if nA1 > 0:
                mA1 = sA['sumx_1'] / nA1

            # Class-conditional variance of each feature in each dataset
            if nA0 > 0:
                vA0 = sA['sum_mean_x2_0'] / nA0 - np.mean(mA0)**2 + eps
            if nA1 > 0:
                vA1 = sA['sum_mean_x2_1'] / nA1 - np.mean(mA1)**2 + eps

            # Average variance of each feature in each dataset
            # (this is to enforze equal variances in both classes)
            if nA0 == 0:
                vA = vA1
            elif nA1 == 0:
                vA = vA0
            else:
                vA = (nA0 * vA0 + nA1 * vA1) / (nA0 + nA1)
            # vB = (nB0 * vB0 + nB1 * vB1) / (nB0 + nB1)

            # Model based on A
            if nA0 == 0:
                w = eps * mA1
                b = 1.0
            elif nA1 == 0:
                w = eps * mA0
                b = - 1.0
            else:
                w = (mA1 - mA0) / vA
                b = (np.log(PA1 / PA0) + 0.5 * np.sum(mA0**2 / vA)
                     - 0.5 * np.sum(mA1**2 / vA))

            # Convert w to column vector
            w = np.array([w]).T

        elif self.mapping == 'spheric':

            eps = 1e-100

            # Sample mean of each feature in each dataset
            mA0, mA1 = 0, 0   # Default values.
            if nA0 > 0:
                mA0 = sA['sumx_0'] / nA0
            if nA1 > 0:
                mA1 = sA['sumx_1'] / nA1

            # Common variance
            sAx2 = sA['sum_mean_x2_0'] + sA['sum_mean_x2_1']
            vA = (sAx2 - np.mean(mA0)**2 * nA0
                       - np.mean(mA1)**2 * nA1) / (nA0 + nA1)

            # Model based on A
            if nA0 == 0:
                w = eps * mA1
                b = 1.0
            elif nA1 == 0:
                w = eps * mA0
                b = - 1.0
            else:
                w = (mA1 - mA0) / vA
                b = (np.log(PA1 / PA0) + 0.5 * np.sum(mA0**2 / vA)
                     - 0.5 * np.sum(mA1**2 / vA))

            # Convert w to column vector
            w = np.array([w]).T

        elif self.mapping == 'scalar':

            # ##################
            # Compute classifier
            eps = 1e-100

            # Sample mean of each feature in each dataset
            mA0, mA1 = 0, 0   # Default values
            if nA0 > 0:
                mA0 = sA['sumx_0'] / nA0
            if nA1 > 0:
                mA1 = sA['sumx_1'] / nA1

            # Class-conditional variance of each feature in each dataset
            vA0, vA1 = 1, 1   # Default values
            if nA0 > 0:
                vA0 = sA['sumx2_0'] / nA0 - mA0**2 + eps
            if nA1 > 0:
                vA1 = sA['sumx2_1'] / nA1 - mA1**2 + eps

            # Average variance of each feature in each dataset
            # (this is to enforze equal variances in both classes)
            if nA0 == 0:
                vA = vA1
            elif nA1 == 0:
                vA = vA0
            else:
                vA = (nA0 * vA0 + nA1 * vA1) / (nA0 + nA1)

            # Model based on A
            if nA0 == 0:
                w = eps * mA1
                b = 1.0
            elif nA1 == 0:
                w = eps * mA0
                b = - 1.0
            else:
                w = (mA1 - mA0) / vA
                b = (np.log(PA1 / PA0) + 0.5 * np.sum(mA0**2 / vA)
                     - 0.5 * np.sum(mA1**2 / vA))

            # Convert w to column vector
            w = np.array([w]).T

        elif self.mapping == 'multidim':

            # ###################
            # Compute classifier
            eps = 1e-10

            # Sample mean of each feature in each dataset
            if nA0 > 0:
                mA0 = np.array([sA['sumx_0']]).T / nA0
            if nA1 > 0:
                mA1 = np.array([sA['sumx_1']]).T / nA1

            # Class-conditional variance of each feature in each dataset
            if nA0 > 0:
                VA0 = sA['Rx_0'] / nA0 - mA0 @ mA0.T
            if nA1 > 0:
                VA1 = sA['Rx_1'] / nA1 - mA1 @ mA1.T

            # Average variance of each feature in each dataset
            # (this is to enforze equal variances in both classes)
            if nA0 == 0:
                VA = VA1
            elif nA1 == 0:
                VA = VA0
            else:
                VA = (nA0 * VA0 + nA1 * VA1) / (nA0 + nA1)

            # Model based on A
            if nA0 == 0:
                w = eps * mA1
                b = 1.0
            elif nA1 == 0:
                w = eps * mA0
                b = - 1.0
            else:
                nf = VA.shape[0]
                inv_VA = np.linalg.inv(VA + eps * np.eye(nf))
                w = inv_VA @ (mA1 - mA0)
                b = (np.log(PA1 / PA0)
                     + 0.5 * mA0.T @ inv_VA @ mA0
                     - 0.5 * mA1.T @ inv_VA @ mA1)

        # ##################
        # Evaluate classifier
        if self.ref == 'paired' and self.mapping == 'spheric':

            mB0 = sB['sumx_0'] / nB0
            mB1 = sB['sumx_1'] / nB1
            vB0 = sB['sum_mean_x2_0'] / nB0 - np.mean(mB0)**2 + eps
            vB1 = sB['sum_mean_x2_1'] / nB1 - np.mean(mB1)**2 + eps

            # Statistics
            z0 = (mB0 @ w + b) / np.sqrt(np.sum(w**2 * vB0))
            z1 = (mB1 @ w + b) / np.sqrt(np.sum(w**2 * vB1))

            # Error probability
            PFA = scpst.norm.cdf(z0[0])
            PM = scpst.norm.cdf(-z1[0])
            Pe = PB0 * PFA + PB1 * PM

        elif self.ref == 'paired' and self.mapping == 'scalar':

            mB0 = sB['sumx_0'] / nB0
            mB1 = sB['sumx_1'] / nB1
            vB0 = sB['sumx2_0'] / nB0 - mB0**2 + eps
            vB1 = sB['sumx2_1'] / nB1 - mB1**2 + eps

            # Statistics
            z0 = (mB0 @ w + b) / np.sqrt((w**2).T @ vB0)
            z1 = (mB1 @ w + b) / np.sqrt((w**2).T @ vB1)

            # Error probability
            PFA = scpst.norm.cdf(z0[0])
            PM = scpst.norm.cdf(-z1[0])
            Pe = PB0 * PFA + PB1 * PM

        elif ((self.ref == 'paired' and self.mapping == 'multidim')
                or self.ref == 'hybrid'):

            eps = 1e-20

            mB0 = np.array([sB['sumx_0']]).T / nB0
            mB1 = np.array([sB['sumx_1']]).T / nB1
            nf = len(mB0)
            VB0 = sB['Rx_0'] / nB0 - mB0 @ mB0.T + eps * np.eye(nf)
            VB1 = sB['Rx_1'] / nB1 - mB1 @ mB1.T + eps * np.eye(nf)

            # Statistics
            z0 = (w.T @ mB0 + b) / np.sqrt(w.T @ VB0 @ w)
            z1 = (w.T @ mB1 + b) / np.sqrt(w.T @ VB1 @ w)

            # NOTE: For consistency with the paired method, the error prob
            # should be computed using a common variance for both classes,
            # as follows. But this does not work well, at least with MNIST
            # VB = (nB0 * VB0 + nB1 * VB1) / (nB0 + nB1)
            # z0 = (w.T @ mB0 + b) / np.sqrt(w.T @ VB @ w)
            # z1 = (w.T @ mB1 + b) / np.sqrt(w.T @ VB @ w)

            # Error probability
            PFA = scpst.norm.cdf(z0[0, 0])
            PM = scpst.norm.cdf(-z1[0, 0])
            Pe = PB0 * PFA + PB1 * PM

        elif self.ref == 'empiric':

            # Statistics
            Pe = np.mean((2 * sB['y'] - 1) * (sB['X'] @ w + b) < 0)

            # This is just to test if a non-equivariance-matrix would work
            # It seems NOT.
            # if self.mapping == 'scalar':
            #     eps = 1e-20
            #     score = (- np.sum(np.log((vA0 + eps) / (vA1 + eps)))
            #              + 2 * np.log((PA1 + eps) / (PA0 + eps))
            #              + np.sum((sB['X'] - mA0)**2 / (vA0 + eps), axis=1)
            #              - np.sum((sB['X'] - mA1)**2 / (vA1 + eps), axis=1))
            #     Pe = np.mean((2 * sB['y'] - 1).T * score < 0)

        # ##################
        # Compute similarity
        sim = 1 - 2 * Pe

        if sim**2 > 1 or np.isnan(sim):
            print("WARNING: similarities are above 1 or nan")
            breakpoint()

        return sim
