from src.data.synthMRWregul import synthMRWregul
from src.data.normalizer import normalize
import src.ctes.str_ctes as sctes
import numpy as np
import argparse


def generate_data(path):
    #%% Data generation for Classication problem

    # We fix two parameters N and win
    LEN_SAMPLE=2**20
    win=1 # large scale window 1 : gaussian, 2 : bump 

    # With N=2**20 we are generating 32 signals of N**15 and L must be <2**16
    N_SUBSAMPLES = 32
    LEN_SUBSAMPLE = int(LEN_SAMPLE / N_SUBSAMPLES)

    # We define the other variables that will change between generations
    c1s=np.array((0.2,0.4,0.6,0.8))
    c2s=np.array((0.02,0.04,0.06,0.08))
    Ls=np.arange(1000,5001,1000)
    epsilons=np.arange(0.5,5.1,1)

    Y_LABELS = [sctes.C1, sctes.C2, sctes.L, sctes.EPSILON]

    mrw=np.zeros((len(c1s), len(c2s), len(Ls), len(epsilons), N_SUBSAMPLES, LEN_SUBSAMPLE + len(Y_LABELS)))

    for ic1 in range(len(c1s)):
        for ic2 in range(len(c2s)):
            for iL in range(len(Ls)):
                for iepsilon in range(len(epsilons)):
                    print(ic1/len(c1s),ic2/len(c2s),iL/len(Ls),iepsilon/len(epsilons))
                    c1=c1s[ic1]
                    c2=c2s[ic2]
                    L=Ls[iL]
                    epsilon=epsilons[iepsilon]
                    x = synthMRWregul(LEN_SAMPLE,c1,c2,L,epsilon,win)
                    y = np.array([c1s[ic1], c2s[ic2], Ls[iL], epsilons[iepsilon]])
                    for isample in range(N_SUBSAMPLES):
                        sub_x = x[isample*LEN_SUBSAMPLE : (isample + 1)*LEN_SUBSAMPLE]
                        mrw[ic1,ic2,iL,iepsilon,isample, :] = np.concatenate([sub_x, y], axis=0)

    XY = mrw.reshape(-1, LEN_SUBSAMPLE + len(Y_LABELS))
    X = XY[:, :LEN_SUBSAMPLE]
    Y = XY[:, LEN_SUBSAMPLE:]

    np.savez(path ,X=X, Y=Y, N_SUBSAMPLES=N_SUBSAMPLES, Y_LABELS=Y_LABELS, 
             c1s=c1s, c2s=c2s, Ls=Ls, epsilons=epsilons)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="data/MRW.npz", help="Path to save data")
    parser.add_argument("--scalerpath", type=str, default="data/scaler.joblib", help="Path to save normalizer")
    args = parser.parse_args()
    generate_data(args.datapath)
    normalize(args.datapath , args.scalerpath)