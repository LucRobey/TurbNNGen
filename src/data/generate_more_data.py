from src.data.generate_data import generate_data
from src.data.normalizer import normalize
import src.ctes.str_ctes as sctes
import numpy as np
import argparse
import tempfile

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, default="data/MRW.npz", help="Path to extend data")
    parser.add_argument("--scalerpath", type=str, default="data/scaler.joblib", help="Path to normalizer")
    args = parser.parse_args()

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmpfile:
        generate_data(tmpfile)
        tmpfilename = tmpfile.name

    normalize(tmpfilename, args.scalerpath, save_scaler=False)
    
    old_data   = dict(np.load(args.datapath))
    new_data   = dict(np.load(tmpfilename))
    merged_data = {}
    merged_data[sctes.X] = np.concatenate((old_data[sctes.X], new_data[sctes.X]), axis=0)
    merged_data[sctes.Y] = np.concatenate((old_data[sctes.Y], new_data[sctes.Y]), axis=0)
    merged_data[sctes.Y_LABELS] = old_data[sctes.Y_LABELS]
    
    np.savez(args.datapath, **merged_data)