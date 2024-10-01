# -*- coding: utf-8 -*-
'''
@author:  Angel Navia Vázquez
OCt. 2024

'''

import numpy as np

class Model():
    
    def __init__(self, NI):
        self.w = np.random.normal(0, 0.001, (NI , 1))

    def fit(self, Xtr_chunks, ytr_chunks, which_workers, Nmaxiter=20, conv_stop=0.005, mu=0.01, Nbatch=1, store_gradients=False, verbose=True):
        """
        Fits the model given the training data. It will only use the data from workers in which_workers.

        Parameters
        ----------
        Xtr_chunks: list of ndarray
            list with input data matrices, one from every worker

        ytr_chunks: list of ndarray
            list with target vectors, one from every worker

        which_workers: list
            indicates which workers must be used

        Returns
        -------
        None (self.w: model weights stored in the model instance)

        """
        if store_gradients:
            self.all_grads_dict = {}
            self.all_w_dict = {}
            self.all_w_dict.update({-1: self.w}) # initial model         
        else:
            self.all_grads_dict = None
            self.all_w_dict = None

        n_iter = 0
        stop_training = False

        NP_dict = {}
        NP_train = 0
        for kworker in which_workers:
            NP_dict.update({kworker: Xtr_chunks[kworker].shape[0]})
            NP_train += Xtr_chunks[kworker].shape[0]
            NI = Xtr_chunks[kworker].shape[1]
        
        if verbose:
            print('\n========== Training the model =========')

        NW = len(Xtr_chunks)

        while not stop_training:

            # Master =======================================
            # broadcasts w to all workers

            # At workers we compute the gradients
            grads_dict = {}  # Gradients at every worker
            for kworker in which_workers:
                w_worker = np.copy(self.w)
                w_worker_old = np.copy(self.w)
                NPtr_worker = Xtr_chunks[kworker].shape[0]
                for nbatch in range(Nbatch):
                    # Compute gradients
                    s = np.dot(Xtr_chunks[kworker], w_worker)
                    o = self.sigm(s)
                    e = o - ytr_chunks[kworker].reshape((-1, 1))
                    grad_worker = np.sum(Xtr_chunks[kworker] * e, axis=0).reshape((NI, 1))
                    w_worker = w_worker - mu * grad_worker / NP_train
                    
                grad_worker = w_worker - w_worker_old
                grads_dict.update({kworker: grad_worker})
            
            # We update weights at Master
            w_old = np.copy(self.w)
            grad_total = np.zeros((NI, 1))
            for kworker in which_workers:
                # Accumulating gradients
                grad_total += grads_dict[kworker]

            #self.w = self.w - mu * grad_total / NP_train

            self.w = self.w  + grad_total

            if store_gradients:
                # We store all the gradients, to be used by the gradient Shapley method    
                self.all_grads_dict.update({n_iter: grads_dict})         
                self.all_w_dict.update({n_iter: self.w})         

            inc_w = np.linalg.norm(self.w - w_old) / np.linalg.norm(w_old)

            n_iter += 1
            # Stop if Maxiter is reached
            if n_iter == Nmaxiter:
                stop_training = True

            if inc_w < conv_stop:
                stop_training = True

            if verbose:
                print('--> iter=%d, inc_w=%f' % (n_iter, inc_w))

        if verbose:
            print('========== Training is complete =========\n')

        if store_gradients:
            if verbose:
                print('The intermediate models are available at model.all_w_dict')
                print('The intermediate gradients are available at model.all_grads_dict')
                print('==========================================\n')


    def sigm(self, x):
        """
        Computes the sigmoid function

        Parameters
        ----------
        x: float
            input value

        Returns
        -------
        sigm(x): float

        """
        return 1 / (1 + np.exp(-x))

    def predict(self, X_b):
        """
        Predicts outputs given the inputs

        Parameters
        ----------
        X_b: ndarray
            Matrix with the input values

        Returns
        -------
        prediction_values: ndarray

        """
        return self.sigm(np.dot(X_b, self.w.ravel()))
