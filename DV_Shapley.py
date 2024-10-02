# -*- coding: utf-8 -*-

'''
"A priori" Estimate Data Values using Shapley. 

@author:  Angel Navia Vázquez
Oct. 2024


'''

import numpy as np
import time, os
import pickle
from Shapley import Shapley
from Models import Models


def sim_cosine(v1, v2):
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


########################################################################
# working directory
working_path = './'
#####################################################################
type_of_features = 'no_ones'
brute_force_model = 'LR'
#####################################################################
# Choose one or more datasets
#####################################################################
datasets = ['mnist_binclass_norm', 'pima', 'income', 'retinopathy', 'spam', 
'cardio', 'w8a', 'news20-1000', 'news20-2000', 'news20-5000', 'news20-10000', 'covtypebin', 
'ijcnn1', 'phishing', 'skin']

datasets = ['pima']

utilities = ['ss', 'rxy_1', 'welch_rxy_t2dif', 'BD_rxy_spheric', 'BD_rxy_scalar', 'Gauss_spheric_paired',  'Gauss_scalar_paired']
utilities = ['Gauss_scalar_paired']


casos = [11] # 5 iid good
casos = [12] # 5 iid good, diferente tamaño
casos = [13] # 3 workers iid good, 2 bad
casos = [16] # 5 workers iid good but with 0%, 20%, 40%, 60%, 80% random targets
casos = [17] # 5 workers iid good but with 0%, 5%, 10%, 15%, 20% flipped targets
casos = [18] # 5 workers iid good but with 0%, 35%, 40%, 45%, 50%, flipped targets
casos = [19] # 5 workers non iid good
casos = [20] # 5 workers iid with noisy inputs: 0, 10, 20, 50, 100 x \sigma_x noise added
casos = [22] # 5 workers, all bad
casos = [23] # 5 workers iid good, input x1, x2, x3, x4, x5
casos = [24] # 5 workers iid good, input x1, x4, x9, x16, x25
casos = [25] # 5 workers iid good, replicating their data x1, x5, x10, x20, x50

casos_paper = [1, 3, 5, 11, 12, 13, 16, 19, 25]
casos = casos_paper

casos = [11]

classes = [0.0, 1.0]

for ref_dataset in ['val', 'tst']:
#for ref_dataset in ['tst']:
    for dataset in datasets: 
        for utility in utilities:
            print('=' * 20)
            print(utility)
            print('=' * 20)

            for caso in casos:
                input_data_path = working_path + 'input_data/%s/' % dataset

                if dataset in ['webspam']:
                    data_file = input_data_path + 'Caso_%d_paper_apriori_val.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xval, yval] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tst.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtst, ytst] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tr1.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtr_chunks, ytr_chunks] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tr2.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtr2, ytr2] = pickle.load(f)
                        Xtr_chunks.append(Xtr2)
                        ytr_chunks.append(ytr2)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tr3a.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtr3a, ytr3a] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tr3b.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtr3b, ytr3b] = pickle.load(f)

                    Xtr3 = np.vstack((Xtr3a, Xtr3b))
                    ytr3 = np.hstack((ytr3a, ytr3b))

                    Xtr_chunks.append(Xtr3)
                    ytr_chunks.append(ytr3)

                elif dataset in ['news20-10000']:
                    data_file = input_data_path + 'Caso_%d_paper_apriori_val.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xval, yval] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tst1.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtst1, ytst1] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tst2.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtst2, ytst2] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tr1.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtr_chunks, ytr_chunks] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tr2.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtr2, ytr2] = pickle.load(f)
                        Xtr_chunks.append(Xtr2)
                        ytr_chunks.append(ytr2)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tr3a.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtr3a, ytr3a] = pickle.load(f)

                    data_file = input_data_path + 'Caso_%d_paper_apriori_tr3b.pkl' % caso
                    with open(data_file, 'rb') as f:
                        [Xtr3b, ytr3b] = pickle.load(f)

                    Xtst = np.vstack((Xtst1, Xtst2))
                    ytst = np.hstack((ytst1, ytst2))

                    Xtr3 = np.vstack((Xtr3a, Xtr3b))
                    ytr3 = np.hstack((ytr3a, ytr3b))

                    Xtr_chunks.append(Xtr3)
                    ytr_chunks.append(ytr3)

                else:
                    data_file = input_data_path + 'Caso_%d_paper_apriori.pkl' % caso
                    print('Loading data from %s...' % data_file)
                    with open(data_file, 'rb') as f:
                        [Xtr_chunks, ytr_chunks, Xval, yval, Xtst, ytst] = pickle.load(f)

                if ref_dataset == 'val':
                    Xref = Xval
                    yref = yval
                if ref_dataset == 'tst':
                    Xref = Xtst
                    yref = ytst

                Nworkers = len(Xtr_chunks)
                which_workers = list(range(Nworkers))

                if utility == 'ACC':
                    V0 = 0.5

                # #####  A PRIORI SHAPLEY ############
                time_ini = time.time()
                print('Computing Shapley values...')

                if utility != 'ACC':
                    # We only compute this for apriori Shapley
                    
                    V0 = 0  # any other value???

                    # Create metric object
                    if utility == 'rxy_1':
                        from Metric_rxy_1 import Metric
                        metric = Metric()
                        # Get a priori statistics from data
                        #ref_stats = metric.get_stats(Xval_b, yval * 2 -1) # avoid targets = 0
                        ref_stats = metric.get_stats(Xref, yref)

                        # Estimating V0 
                        V0s = []
                        for k in range(50):
                            # Estimating V0
                            #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                            X = Xref
                            y = (np.random.normal(0, 1, yref.shape) > 0).astype(float)
                            #y = y * 2 -1
                            tmp_metric = Metric()
                            random_stats = tmp_metric.get_stats(X, y)
                            V0s.append(tmp_metric.S(ref_stats, random_stats))  

                        V0 = np.mean(V0s)

                    elif utility == 'rxy_avg':
                        from Metric_rxy_avg import Metric
                        metric = Metric(classes)
                        # Get a priori statistics from data
                        ref_stats = [metric.get_stats(Xref, yref)]

                        # Estimating V0 
                        V0s = []
                        for k in range(50):
                            # Estimating V0
                            #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                            X = Xref
                            y = (np.random.normal(0, 1, yref.shape) > 0).astype(float)
                            #y = y * 2 -1
                            tmp_metric = Metric(classes)
                            random_stats = [tmp_metric.get_stats(X, y)]
                            V0s.append(tmp_metric.S(ref_stats, random_stats))  

                        V0 = np.mean(V0s)

                    elif utility == 'rxy_mix':
                        unbalance = []
                        Nworkers = len(ytr_chunks)
                        for kworker in range(Nworkers):
                            y = ytr_chunks[kworker]
                            N0 = np.sum(y==0)  
                            N1 = np.sum(y==1)
                            bal = (np.max([N0, N1]) -  np.min([N0, N1])) / (N0 + N1)
                            print(kworker, bal)
                            unbalance.append(bal)

                        UV = np.max(unbalance)

                        if UV > 0.5:
                            from Metric_rxy_avg import Metric
                            metric = Metric(classes)
                            # Get a priori statistics from data
                            ref_stats = [metric.get_stats(Xref, yref)]

                            # Estimating V0 
                            V0s = []
                            for k in range(50):
                                # Estimating V0
                                #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                                X = Xref
                                y = (np.random.normal(0, 1, yref.shape) > 0).astype(float)
                                #y = y * 2 -1
                                tmp_metric = Metric(classes)
                                random_stats = [tmp_metric.get_stats(X, y)]
                                V0s.append(tmp_metric.S(ref_stats, random_stats))  

                            V0 = np.mean(V0s)

                        else:
                            from Metric_rxy_1 import Metric
                            metric = Metric()
                            # Get a priori statistics from data
                            #ref_stats = metric.get_stats(Xval_b, yval * 2 -1) # avoid targets = 0
                            ref_stats = metric.get_stats(Xref, yref)

                            # Estimating V0 
                            V0s = []
                            for k in range(50):
                                # Estimating V0
                                #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                                X = Xref
                                y = (np.random.normal(0, 1, yref.shape) > 0).astype(float)
                                #y = y * 2 -1
                                tmp_metric = Metric()
                                random_stats = tmp_metric.get_stats(X, y)
                                V0s.append(tmp_metric.S(ref_stats, random_stats))  

                            V0 = np.mean(V0s)

                    elif utility == 'rxy_hybrid':
                        from Metric_rxy_hybrid import Metric
                        metric = Metric(classes)
                        # Get a priori statistics from data
                        ref_stats = [metric.get_stats(Xref, yref)]

                        # Estimating V0 
                        V0s = []
                        for k in range(50):
                            # Estimating V0
                            #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                            X = Xref
                            y = (np.random.normal(0, 1, yref.shape) > 0).astype(float)
                            #y = y * 2 -1
                            tmp_metric = Metric(classes)
                            random_stats = [tmp_metric.get_stats(X, y)]
                            V0s.append(tmp_metric.S(ref_stats, random_stats))  

                        V0 = np.mean(V0s)

                    elif utility == 'rxy_hybrid2':
                        from Metric_rxy_hybrid2 import Metric

                        unbalance = []
                        Nworkers = len(ytr_chunks)
                        for kworker in range(Nworkers):
                            y = ytr_chunks[kworker]
                            N0 = np.sum(y==0)  
                            N1 = np.sum(y==1)
                            bal = (np.max([N0, N1]) -  np.min([N0, N1])) / (N0 + N1)
                            print(kworker, bal)
                            unbalance.append(bal)

                        UV = np.max(unbalance)

                        metric = Metric(classes, UV)
                        # Get a priori statistics from data
                        ref_stats = [metric.get_stats(Xref, yref)]

                        # Estimating V0 
                        V0s = []
                        for k in range(50):
                            # Estimating UV
                            #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                            X = Xref
                            y = (np.random.normal(0, 1, yref.shape) > 0).astype(float)
                            tmp_metric = Metric(classes, UV)
                            random_stats = [tmp_metric.get_stats(X, y)]
                            V0s.append(tmp_metric.S(ref_stats, random_stats))  

                        V0 = np.mean(V0s)
                        metric.V0 = V0

                    elif utility == 'meanstd':
                        from Metric_meanstd import Metric
                        metric = Metric()
                        # Get a priori statistics from data
                        ref_stats = metric.get_stats(Xref, yref)

                        # Estimating V0 
                        V0s = []
                        for k in range(50):
                            # Estimating V0
                            #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                            X = Xref
                            y = (np.random.normal(0, 1, yref.shape) > 0).astype(float)
                            tmp_metric = Metric()
                            random_stats = tmp_metric.get_stats(X, y)
                            V0s.append(tmp_metric.S(ref_stats, random_stats))  

                        V0 = np.mean(V0s)

                    elif utility == 'ss':
                        from Metric_ss import Metric
                        metric = Metric()
                        # Get a priori statistics from data
                        ref_stats = metric.get_stats(Xref, yref)

                        # Estimating V0 
                        V0s = []
                        for k in range(50):
                            # Estimating V0
                            #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                            X = Xref
                            y = (np.random.normal(0, 1, yref.shape) > 0).astype(float)
                            tmp_metric = Metric()
                            random_stats = tmp_metric.get_stats(X, y)
                            V0s.append(tmp_metric.S(ref_stats, random_stats))  

                        V0 = np.mean(V0s)

                    elif utility[:5] == 'welch':
                        from Metric_welch import Metric
                        # Take the metric options from the utility name
                        f = utility.split('_')[1]
                        m = utility.split('_')[2]
                        metric = Metric(feature_map=f, mapping=m)
                        # Get a priori statistics from data
                        ref_stats = metric.get_stats(Xref, yref)

                    elif utility[:2] == 'BD':

                        from Metric_BD import Metric
                        # Take the metric options from the utility name
                        if type_of_features == 'ones':
                            f = 'rxy'
                        else:
                            f = 'rx1y'
                        m = utility.split('_')[2]
                        metric = Metric(feature_map=f, mapping=m)
                        # Get a priori statistics from data
                        ref_stats = metric.get_stats(Xref, yref)

                    elif utility[:5] == 'Gauss':

                        from Metric_Gauss import Metric
                        # Take the metric options from the utility name
                        m = utility.split('_')[1]
                        ref = utility.split('_')[2]
                        metric = Metric(option=type_of_features, mapping=m, ref=ref)
                        # Get a priori statistics from data
                        ref_stats = metric.get_stats(Xref, yref)

                    elif utility == 'rxy_span':
                        from Metric_rxy_span import Metric
                        metric = Metric()
                        # Get a priori statistics from data
                        ref_stats = metric.get_ref_stats(Xref, yref * 2 - 1)         

                        # Estimating V0 
                        V0s = []
                        for k in range(50):
                            # Estimating V0
                            #X = np.random.normal(np.mean(Xval_b), np.std(Xval_b), Xval_b.shape)
                            X = Xref
                            y = (np.random.normal(0, 1, yref.shape) > 0.5).astype(float)
                            y = y * 2 - 1
                            tmp_metric = Metric()

                            random_stats = tmp_metric.get_stats(X, y)
                            #V0s.append(sim_cosine(np.array(random_stats['rxy_span']),np.array(ref_stats['rxy_span'])))  
                            V0s.append(tmp_metric.S_span(random_stats, ref_stats))  

                        V0 = np.mean(V0s)

                    # Computing stats for workers
                    NW = len(Xtr_chunks)
                    which_workers = list(range(NW))
                    worker_stats_dict = {}
                    for kworker in which_workers:
                        worker_stats_dict.update({kworker: metric.get_stats(
                            Xtr_chunks[kworker], ytr_chunks[kworker])})

                    # Training models
                    models = Models(Xtr_chunks, ytr_chunks, Xref, yref,
                                    worker_stats_dict, ref_stats, metric)
                    models_dict = models.get_models(utility)

                if utility == 'ACC':
                    # #####  BRUTE FORCE SHAPLEY ############
                    # Training models
                    models = Models(Xtr_chunks, ytr_chunks, Xref, yref, brute_force_model, brute_force_model=brute_force_model)
                    models_dict = models.get_models(utility)

                # Computing Shapley values in either brute force or apriori
                shapley = Shapley(V0)
                PHIs = shapley.compute_scores(which_workers, models_dict, verbose=False)

                print('=' * 50)
                print('Data value of each worker:')
                print('=' * 50)
                print('Caso = %d, Cost=%s' % (caso, utility))

                DVnorm = np.copy(PHIs)
                # Warning, some PHIs are negative, we eliminate those values in DVnorm
                DVnorm[DVnorm < 0] = 0
                # We normalize DV to sum 1 when the values are significative, otherwise all workers are bad
                if np.sum(DVnorm) > 0:
                    DVnorm = DVnorm / np.sum(DVnorm)

                if np.sum(DVnorm) == 0:
                    DVnorm = np.ones(DVnorm.shape)
                    DVnorm = DVnorm / np.sum(DVnorm)      

                for kselected in which_workers:
                    print('%d -> %f (%f)' % (
                        which_workers[kselected] + 1, DVnorm[kselected], PHIs[kselected]))
                print('=' * 50)

                time_end = time.time()
                training_time = float((time_end - time_ini) / 60.0)
                print('Elapsed minutes = %f' % training_time)
                print('=' * 50)

                try: 
                    AUs = shapley.AUs
                except:
                    AUs = None

                w_ = [str(ww) for ww in which_workers]
                wnames = '_'.join(w_)

                output_data_path = working_path + 'results/%s/%s/'%(dataset, ref_dataset)
                # checking output folders:
                if not os.path.exists(output_data_path):
                    os.makedirs(output_data_path)

                if utility == 'ACC': # #####  BRUTE FORCE SHAPLEY ############
                    filename =  output_data_path + f'Shapley_caso_{caso}_{utility}_{brute_force_model}_workers_{wnames}.pkl'
                else:
                    filename = output_data_path + f'Shapley_caso_{caso}_{utility}_workers_{wnames}.pkl'

                with open(filename, 'wb') as f:
                    pickle.dump([which_workers, PHIs, DVnorm, models_dict,
                                 training_time, shapley.contribs, AUs], f)

                print('Saved results in %s' % filename)
