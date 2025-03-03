import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from src.CDMD_dataset import CDMDDataset
from src.mlf_alpha_estimator import estimate_mlf_alpha


if __name__ == "__main__":
    dataset_root = Path("../CDMD")
    mlf_fit_path = Path("../CDMD/results/mlf_alpha_fit_fix.npy")
    big_delta = 19
    slice_id = 50

    dataset = CDMDDataset(dataset_root)
    data, mask = dataset.load_sample("sub_005", big_delta, slice_id)

    if not mlf_fit_path.exists:
        results = estimate_mlf_alpha(data, mask, big_delta, dataset.small_delta)
        np.save(mlf_fit_path, results)
    else:
        results = np.load(mlf_fit_path)
    
    D, alpha, err = cv2.split(results)
    plt.imshow(alpha, cmap="gray")
    plt.show()
