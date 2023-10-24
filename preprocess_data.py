import numpy as np
import ctes

def preprocess_data(pre_mrw_path, mrw_path):
    data = np.load(pre_mrw_path)

    mrw = data[ctes.MRW]

    c1s = data[ctes.C1S]
    c2s = data[ctes.C2S]
    Ls = data[ctes.LS]
    epsilons = data[ctes.EPSILONS]

    N = data[ctes.N]
    win = data[ctes.WIN]

    S_LABELS = [ctes.L, ctes.ETA,  ctes.H, ctes.C1]

    S = np.zeros((list(mrw.shape[:-1]) + [len(S_LABELS)]))
    
    for ic1 in range(mrw.shape[0]):
        for ic2 in range(mrw.shape[1]):
            for iL in range(mrw.shape[2]):
                for iepsilon in range(mrw.shape[3]):
                    print(ic1/len(c1s),ic2/len(c2s),iL/len(Ls),iepsilon/len(epsilons))
                    stats = {}
                    stats[ctes.L] = Ls[iL]
                    stats[ctes.ETA] = epsilons[iepsilon]
                    stats[ctes.H] = c1s[ic1] + c2s[ic2] 
                    stats[ctes.C1] = (-1) * c2s[ic2]

                    S[ic1, ic2, iL, iepsilon] = np.array([stats[label] for label in S_LABELS])

    X = mrw.reshape(-1, N)
    S = S.reshape(-1, len(S_LABELS))

    np.savez(mrw_path, X=X, S=S, S_LABELS=S_LABELS, N=N, win=win)


if __name__ == "__main__":
    preprocess_data("Pre_MRW.npz", "MRW.npz")




