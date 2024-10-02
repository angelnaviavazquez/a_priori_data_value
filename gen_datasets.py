# -*- coding: utf-8 -*-

'''
@author:  Angel Navia Vázquez
Oct. 2024

Generates the different benchmark cases datasets 

'''

import numpy as np
import time
import pickle
import os


seed = 1234
np.random.seed(seed=seed)
working_path = './'

casos = [1, 3, 5, 11, 12, 13, 16, 19, 25]
# Use this to generate a single/new case 
casos = [11]

datasets = ['mnist_binclass_norm', 'pima', 'income', 'retinopathy', 'spam', 
'cardio', 'w8a', 'news20-1000', 'news20-2000', 'news20-5000', 'news20-10000', 'covtypebin', 
'ijcnn1', 'phishing', 'skin']
# Use this to operate on a single dataset 
datasets = ['pima']

for dataset in datasets: 
    input_data_path = working_path + 'input_data/'
    output_data_path = input_data_path + dataset + '/'

    ### DELETE THIS
    '''
    data_file = input_data_path + dataset + '_demonstrator_data.pkl'
    print('Loading data from %s...' %  data_file)
    with open(data_file, 'rb') as f:
        [Xtr_chunks, ytr_chunks, Xval, yval, Xtst, ytst] = pickle.load(f)

    Xtr = np.vstack(Xtr_chunks)
    ytr = np.hstack(ytr_chunks)

    data_file = input_data_path + dataset + '_data.pkl'
    print('Saving data to  %s...' %  data_file)
    with open(data_file, 'wb') as f:
        pickle.dump([Xtr, ytr, Xval, yval, Xtst, ytst], f)
    '''

    data_file = input_data_path + dataset + '_data.pkl'
    print('Loading data from %s...' %  data_file)
    with open(data_file, 'rb') as f:
        [Xtr, ytr, Xval, yval, Xtst, ytst] = pickle.load(f)


    for caso in casos:
        print('Dataset = %s, Caso = %d'%(dataset, caso))

        NPtr = Xtr.shape[0]
        ind = np.random.permutation(NPtr)
        Xtr = Xtr[ind, :]
        ytr = ytr[ind]

        Xtr_chunks = []
        ytr_chunks = []

        # All cases have 5 workers
        which_workers = [0, 1, 2, 3, 4]
        Nworkers = len(which_workers)
        Nchunk = int(NPtr/Nworkers)

        if caso == 11: # 5 workers iid good
            texttitle = '5 workers iid good'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                print(Xtr_.shape)
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

        if caso == 12: # 5 workers different datasize 40%, 30%, 20%, 9%, 1%
            texttitle = '5 workers different datasize'

            # worker 0
            Xtr_ = Xtr[0: int(0.40 * NPtr), :]
            ytr_ = ytr[0: int(0.40 * NPtr)]
            print(Xtr_.shape, ytr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

            # worker 1
            Xtr_ = Xtr[int(0.40 * NPtr): int(0.7 * NPtr), :]
            ytr_ = ytr[int(0.40 * NPtr): int(0.7 * NPtr)]
            ## repmat x10
            #Xtr_ = np.kron(np.ones((10, 1)), Xtr_)
            #ytr_ = np.kron(np.ones((1, 10)), ytr_).ravel()
            print(Xtr_.shape, ytr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

            # worker 2
            Xtr_ = Xtr[int(0.7 * NPtr): int(0.90 * NPtr), :]
            ytr_ = ytr[int(0.7 * NPtr): int(0.90 * NPtr)]
            print(Xtr_.shape, ytr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

            # worker 3
            Xtr_ = Xtr[int(0.90 * NPtr): int(0.99 * NPtr), :]
            ytr_ = ytr[int(0.90 * NPtr): int(0.99 * NPtr)]
            print(Xtr_.shape, ytr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

            # worker 4
            Xtr_ = Xtr[int(0.99 * NPtr): , :]
            ytr_ = ytr[int(0.99 * NPtr): ]
            print(Xtr_.shape, ytr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

        if caso == 13: # 3 workers iid good, 2 bad
            texttitle = '3 workers iid good, 2 bad'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                print(Xtr_.shape)
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

            mx = np.mean(Xtr)
            stdx = np.std(Xtr)

            # Workers 3 and 4 with random data
            Xtr_chunks[3] = np.random.normal(mx, stdx, Xtr_chunks[3].shape)
            ytr_chunks[3] = (np.random.normal(0, 1, ytr_chunks[3].shape) > 0).astype(float)
            Xtr_chunks[4] = np.random.normal(mx, stdx, Xtr_chunks[4].shape)
            ytr_chunks[4] = (np.random.normal(0, 1, ytr_chunks[4].shape) > 0).astype(float)


        if caso == 16: # 
            texttitle = '5 workers iid good but with 0%, 20%, 40%, 60%, 80% random targets'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                print(Xtr_.shape)
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

            # 20% random
            pos = 1
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NR = int(0.2 * N)
            yr = (np.sign(np.random.normal(0, 1, (NR, 1))).ravel() + 1 ) / 2
            y[0: NR] = yr
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 40% random
            pos = 2
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NR = int(0.4 * N)
            yr = (np.sign(np.random.normal(0, 1, (NR, 1))).ravel() + 1 ) / 2
            y[0: NR] = yr
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 60% random
            pos = 3
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NR = int(0.6 * N)
            yr = (np.sign(np.random.normal(0, 1, (NR, 1))).ravel() + 1 ) / 2
            y[0: NR] = yr
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 80% random
            pos = 4
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NR = int(0.8 * N)
            yr = (np.sign(np.random.normal(0, 1, (NR, 1))).ravel() + 1 ) / 2
            y[0: NR] = yr
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y


        if caso == 17: # 5 workers iid good
            texttitle = '5 workers iid good but with 0%, 5%, 10%, 15%, 20% flipped targets'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                print(Xtr_.shape)
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

            # 5% flip
            pos = 1
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NF = int(0.05 * N)
            yf = 1 - y[0: NF]
            y[0: NF] = yf
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 10% flip
            pos = 2
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NF = int(0.1 * N)
            yf = 1 - y[0: NF]
            y[0: NF] = yf
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 15% flip
            pos = 3
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NF = int(0.15 * N)
            yf = 1 - y[0: NF]
            y[0: NF] = yf
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 20% flip
            pos = 4
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NF = int(0.20 * N)
            yf = 1 - y[0: NF]
            y[0: NF] = yf
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

        if caso == 18: # 5 workers iid good
            texttitle = '5 workers iid good but with 0%, 35%, 40%, 45%, 50%, flipped targets'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                print(Xtr_.shape)
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

            # 20% flip
            pos = 1
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NF = int(0.35 * N)
            yf = 1 - y[0: NF]
            y[0: NF] = yf
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 30% flip
            pos = 2
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NF = int(0.4 * N)
            yf = 1 - y[0: NF]
            y[0: NF] = yf
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 40% flip
            pos = 3
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NF = int(0.45 * N)
            yf = 1 - y[0: NF]
            y[0: NF] = yf
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

            # 60% flip
            pos = 4
            X = Xtr_chunks[pos]
            y = ytr_chunks[pos] 
            N = len(y)
            ind = np.random.permutation(N)
            y = y[ind]
            X = X[ind, :]
            NF = int(0.5 * N)
            yf = 1 - y[0: NF]
            y[0: NF] = yf
            Xtr_chunks[pos] = X
            ytr_chunks[pos] = y

        if caso == 19: # 5 workers non iid good
            texttitle = '5 workers non iid good'
            which_workers = [0, 1, 2, 3, 4]

            which0 = ytr == 0
            which1 = ytr == 1
            Xtr0 = Xtr[which0, :]
            ytr0 = ytr[which0]
            NPtr0 = len(ytr0)
            Xtr1 = Xtr[which1, :]
            ytr1 = ytr[which1]
            NPtr1 = len(ytr1)

            Xtr_chunks = []
            ytr_chunks = []
            Nchunk0 = int(NPtr0/10)
            Nchunk1 = int(NPtr1/10)

            ind = np.random.permutation(Xtr0.shape[0])
            Xtr0 = Xtr0[ind, :]
            ytr0 = ytr0[ind]

            ind = np.random.permutation(Xtr1.shape[0])
            Xtr1 = Xtr1[ind, :]
            ytr1 = ytr1[ind]

            Xtr_4_0 = Xtr0[ 0 * Nchunk0: (0 + 4) * Nchunk0 + 1, :]
            ytr_4_0 = ytr0[ 0 * Nchunk0: (0 + 4) * Nchunk0 + 1]

            Xtr_3_0 = Xtr0[ 4 * Nchunk0: (4 + 3) * Nchunk0 + 1, :]
            ytr_3_0 = ytr0[ 4 * Nchunk0: (4 + 3) * Nchunk0 + 1]

            Xtr_2_0 = Xtr0[ 7 * Nchunk0: (7 + 2) * Nchunk0 + 1, :]
            ytr_2_0 = ytr0[ 7 * Nchunk0: (7 + 2) * Nchunk0 + 1]

            Xtr_1_0 = Xtr0[ 9 * Nchunk0: , :]
            ytr_1_0 = ytr0[ 9 * Nchunk0: ]

            Xtr_4_1 = Xtr1[ 0 * Nchunk1: (0 + 4) * Nchunk1 + 1, :]
            ytr_4_1 = ytr1[ 0 * Nchunk1: (0 + 4) * Nchunk1 + 1]

            Xtr_3_1 = Xtr1[ 4 * Nchunk1: (4 + 3) * Nchunk1 + 1, :]
            ytr_3_1 = ytr1[ 4 * Nchunk1: (4 + 3) * Nchunk1 + 1]

            Xtr_2_1 = Xtr1[ 7 * Nchunk1: (7 + 2) * Nchunk1 + 1, :]
            ytr_2_1 = ytr1[ 7 * Nchunk1: (7 + 2) * Nchunk1 + 1]

            Xtr_1_1 = Xtr1[ 9 * Nchunk1: , :]
            ytr_1_1 = ytr1[ 9 * Nchunk1: ]

            Xtr_ = Xtr_4_0
            ytr_ = ytr_4_0
            print(Xtr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

            Xtr_ = Xtr_4_1
            ytr_ = ytr_4_1
            print(Xtr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

            Xtr_ = np.vstack(( Xtr_3_0 , Xtr_1_1 ))
            ytr_ = np.hstack(( ytr_3_0 , ytr_1_1 ))
            print(Xtr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

            Xtr_ = np.vstack(( Xtr_1_0 , Xtr_3_1 ))
            ytr_ = np.hstack(( ytr_1_0 , ytr_3_1 ))
            print(Xtr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

            Xtr_ = np.vstack(( Xtr_2_0 , Xtr_2_1 ))
            ytr_ = np.hstack(( ytr_2_0 , ytr_2_1 ))
            print(Xtr_.shape)
            Xtr_chunks.append(Xtr_)
            ytr_chunks.append(ytr_)

        if caso == 20: # 
            texttitle = '5 workers iid with noisy inputs: 0, 10, 20, 50, 100 x \sigma_x noise added'
            sigma_x = np.std(Xtr)
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                if kworker == 0:
                    factor = 0
                if kworker == 1:
                    factor = 10
                if kworker == 2:
                    factor = 20
                if kworker == 3:
                    factor = 50
                if kworker == 4:
                    factor = 100

                noise = np.random.normal(0, 2 * sigma_x, Xtr_.shape) 
                Xtr_ += noise

                print(Xtr_.shape)
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

        if caso == 22: # 5 workers bad
            texttitle = '5 workers, all bad'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                print(Xtr_.shape)

                Xtr_rand = np.random.normal(0, 1, Xtr_.shape)
                Xtr_chunks.append(Xtr_rand)
                NR = ytr_.shape[0]
                yr = (np.random.normal(0, 1, (NR, 1)) > 0).ravel().astype(float)
                ytr_chunks.append(yr)

        if caso == 23: # 5 workers iid good, with inputs multiplied 
            texttitle = '5 workers iid good, input x1, x2, x3, x4, x5'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                print(Xtr_.shape)
                Xtr_ = Xtr_ * (kworker + 1)
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

        if caso == 24: # 5 workers iid good, with inputs multiplied 
            texttitle = '5 workers iid good, input x1, x4, x9, x16, x25'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]
                print(Xtr_.shape)
                Xtr_ = Xtr_ * (kworker + 1) ** 2
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

        if caso == 25: # 5 workers iid good, pero con repmat
            texttitle = '5 workers iid good, replicating their data x1, x5, x10, x20, x50'
            for kworker in which_workers:
                if kworker == Nworkers - 1: # the last one
                    Xtr_ = Xtr[kworker*Nchunk:, :]
                    ytr_ = ytr[kworker*Nchunk:]
                else:
                    Xtr_ = Xtr[kworker*Nchunk:(kworker+1)*Nchunk, :]
                    ytr_ = ytr[kworker*Nchunk:(kworker+1)*Nchunk]

                if kworker == 0:
                    factor = 1
                if kworker == 1:
                    factor = 5
                if kworker == 2:
                    factor = 10
                if kworker == 3:
                    factor = 20
                if kworker == 4:
                    factor = 50

                ## repmat
                Xtr_ = np.kron(np.ones((factor, 1)), Xtr_)
                ytr_ = np.kron(np.ones((1, factor)), ytr_).ravel()

                print(Xtr_.shape)
                Xtr_chunks.append(Xtr_)
                ytr_chunks.append(ytr_)

        # checking output folders:
        if not os.path.exists(output_data_path):
            os.makedirs(output_data_path)

        data_file = output_data_path + 'Caso_%d_paper_apriori.pkl' % caso
        with open(data_file, 'wb') as f:
            pickle.dump([Xtr_chunks, ytr_chunks, Xval, yval, Xtst, ytst], f)
            
        print('Saved data to %s...' %  data_file)

