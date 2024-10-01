# -*- coding: utf-8 -*-
'''

Class that defines the metric to be used in the A priori Shapley estimation

@author:  Jesús Cid Sueiro
Oct. 2024

'''

import numpy as np
from scipy.stats import t


class Metric():

    def __init__(self, feature_map='rx1y', mapping='linear'):
        """
        Parameters
        ----------
        feature_map: str, optional (default='standard')
            Type of feature mapping: 'standard', 'rxy', 'addxy'
        mapping: None or str in {'linear', 't'}, optional (defaul='linear')
            Type of mapping of the t-statistic into a [0, 1] interval.
        """

        self.feature_map = feature_map
        self.mapping = mapping

        self.name = f'welch_{self.feature_map}_{self.mapping}'

    def get_features(self, X, y):
        """
        Compute the feature matrix for the computation of statistics:
        """

        X = X.astype(float)
        # We map targets to (-1, 1)
        y = np.array(y).astype(float).reshape(-1, 1) * 2 - 1

        if self.feature_map == 'standard':
            Z = np.hstack((X, y))

        elif self.feature_map == 'rxy':
            Z = X * y

        elif self.feature_map == 'rx1y':
            # This is similar to rxy, but it add an all-ones colum to CX, but
            # it does not make a difference (in my experiment)
            # ones = np.ones((X.shape[0], 1))
            # Z = np.hstack((X, ones)) * y
            # Add an all ones colum to CX
            ones = np.ones((X.shape[0], 1))
            Z = np.hstack((X, ones)) * y

        elif self.feature_map == 'addxy':
            Z = np.hstack((X, y, X * y))

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

        stats = {'N': Z.shape[0],
                 'sumz': np.sum(Z, axis=0),
                 'sumz2': np.sum(Z**2, axis=0)}

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

        # Dataset sizes
        nA = sA['N']
        nB = sB['N']

        # Sample mean of each feature in each dataset
        mA = sA['sumz'] / nA
        mB = sB['sumz'] / nB
        # Variance of each feature in each dataset
        vA = sA['sumz2'] / nA - mA**2
        vB = sB['sumz2'] / nB - mB**2

        # This is just to be a bit more precise:
        vA = vA * nA / (nA - 1)
        vB = vB * nB / (nB - 1)

        # Note that t2 is a value in [0, inf]. The higher it is, the smaller
        # the similarity is. Therefore, we should compute the similarity by
        # some monotonically decreasing transformation:
        if self.mapping == 'linear':
            # Compute t-value
            # Taking into account that den2 > s2B**2 / nB
            rA = vA / nA
            rB = vB / nB
            t2 = (mA - mB)**2 / (rA + rB + eps)

            # Compute upper bound on t2,for a fixed reference B and fixed mxA
            t2_max = (mA - mB)**2 / (rB + eps)

            # Normalize similarity
            # t2 is a value in [0, t2_max]. The higher it is, the smaller the
            # similarity is.
            # We compute the similarity by a linearly decreasing transformation
            sims = 1 - t2 / t2_max
            sim = np.mean(sims)

        elif self.mapping == 't':
            # The following is based in the t-test for evaluating if two
            # populations have the same means.
            # We adapt the welch-test, which does not assume equal variances
            # See https://en.wikipedia.org/wiki/Student%27s_t-test
            # The test is used for each feature independently.
            # For a more grounde extension to the multidimensional case see
            # https://en.wikipedia.org/
            # wiki/Hotelling%27s_T-squared_distribution#Two-sample_statistic

            # t values
            rA = vA / nA
            rB = vB / nB

            # Positive t-values, to test if mean(A) = mean(B)
            t_pos = np.sqrt((mA - mB)**2 / (rA + rB + eps))
            # Negative t-values, to test if mean(A) = mean(-B)
            # This is to test the similarity between A and the result of
            # flipping the labels in B.
            # (The code assumes that self.feature_map = 'rxy', so that the
            #  sample mean of the reference B after flipping y by -y is -mB)
            t_neg = np.sqrt((mA + mB)**2 / (rA + rB + eps))

            # (Note that both t_pos and t_neg contain non-negative values.
            #  pos and neg refer to the true and the flipped labels, resp.)

            # Degrees of freedom
            nu = (rA + rB)**2 / (rA**2 / (nA - 1) + rB**2 / (nB - 1) + eps)
            nu += eps     # To avoid zero divisions

            # t statistics
            # This are the pdfs of the t values. Not used
            # sims = (1 + t_pos**2 / nu)**(-(nu + 1) / 2)
            # sims_neg = (1 + t_neg**2 / nu)**(-(nu + 1) / 2)
            # For large nu, we should have sims close to np.exp(-t_pos**2 / 2)

            # Similarity.
            # The similarity between A and B is computed as the cdf at t
            # That is, S_pos is the probability that r.v. T is higher (in
            # absolute value) than the observed value, t_pos, given the null
            # hypothesis (mA==mB),
            # S_pos = 2 * (1 - t.cdf(t_pos, nu))
            # S_neg = 2 * (1 - t.cdf(t_neg, nu))
            # We are interested in the difference, that is
            D = 2 * (t.cdf(t_neg, nu) - t.cdf(t_pos, nu))

            # The similarity value will be computed as sim = D / Dmax
            # where r is just a scale factor so that:
            # - If mA = mB, we should get sim == 1
            # - If mA = -mB, we should get sim == -1.
            # Since, for mA == mB, we get t_pos == 0 and
            #     t_neg = np.sqrt((2 * mB)**2 / (rA + rB + eps))
            #     D = 2 * t.cdf(t_neg, nu)
            # and, by symmetry, for mA == -mB, we get t_neg == 0 and
            #     t_pos = np.sqrt((2 * mB)**2 / (rA + rB + eps))
            #     D = - 2 * t.cdf(t_pos, nu)
            # we can take
            Dmax = 2 * t.cdf(np.sqrt((2 * mB)**2 / (rA + rB + eps)), nu)
            # and
            D = D / Dmax

            # # Chisquare option
            # dim = len(t2)
            # c = np.sum(t2)
            # sim_chi = 1 - chi2.cdf(c, dim)
            # c_neg = np.sum(t2_neg)
            # sim_neg_chi = 1 - chi2.cdf(c_neg, dim)
            # simc = sim_chi - sim_neg_chi

            # The final similarity is the average of the normalized similarity
            # values
            sim = np.mean(D)

        elif self.mapping == 't2dif':

            # t values
            rA = vA / nA
            rB = vB / nB

            # TESTING (a revisited welch). The following is a version similar
            # to this:
            # tmax2 = (2 * mB)**2 / (rA + rB + eps) + eps
            # D = (np.sum(t_neg**2) - np.sum(t_pos**2)) / np.sum(tmax2)

            mAnorm = mA / (np.sqrt(rA + rB + eps) + eps)
            mBnorm = mB / (np.sqrt(rA + rB + eps) + eps)

            d = mAnorm.T @ mBnorm
            vAnorm = np.sqrt(mAnorm.T @ mAnorm)
            vBnorm = np.sqrt(mBnorm.T @ mBnorm)
            sim = d / (vAnorm * vBnorm + eps)

            return sim

        elif self.mapping == 'tlim':
            # This is a variant of self.m that assumes that the statistics of
            # the reference distribution B are exact. This is like taking
            # nB == infinity.

            # t values
            rA = vA / nA

            # Positive t-values, to test if mean(A) = mean(B)
            t_pos = np.sqrt((mA - mB)**2 / (rA + eps))
            # Negative t-values, to test if mean(A) = mean(-B)
            t_neg = np.sqrt((mA + mB)**2 / (rA + eps))

            # (Note that both t_pos and t_neg contain non-negative values.
            #  pos and neg refer to the true and the flipped labels, resp.)

            # Degrees of freedom
            nu = nA - 1
            nu += eps     # To avoid divisions by 0 in degenerate cases

            # Similarity.
            D = 2 * (t.cdf(t_neg, nu) - t.cdf(t_pos, nu))
            Dmax = 2 * t.cdf(np.sqrt((2 * mB)**2 / (rA + eps)), nu)
            D = D / Dmax

            # The final similarity is the average of the normalized similarity
            # values
            sim = np.mean(D)

        elif self.mapping == 'cosine':
            # WARNING: Next lines assume vA and vB 1D arrays or column vectors
            d = mA.T @ mB
            vAnorm = np.sqrt(mA.T @ mA)
            vBnorm = np.sqrt(mB.T @ mB)
            sim = d / (vAnorm * vBnorm + eps)

            # breakpoint()
            # a = 1

        else:
            sim = -t

        # v1 = self.get_vector(s1)
        # v2 = self.get_vector(s2)
        # return self.sim_cosine(v1, v2).ravel()[0]
        return sim
