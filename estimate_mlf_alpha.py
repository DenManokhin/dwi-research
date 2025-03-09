import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.CDMD_dataset import CDMDDataset
from src.mlf_alpha_estimator import estimate_mlf_alpha,\
    compute_entropy, compute_kurtosis
from src.util import save_results, load_results, plot_results,\
    exclude_outliers


if __name__ == "__main__":
    big_delta = 19
    slice_id = 50
    sample_name = "sub_005"
    dataset_root = Path("../CDMD")
    results_dir = Path("../CDMD/results_mlf_alpha/") / sample_name / str(slice_id)

    dataset = CDMDDataset(dataset_root)
    sample = dataset.load_sample(sample_name, big_delta, slice_id)
    results = load_results(results_dir)

    if not ("D_MLF" in results and "Alpha" in results):
        results = estimate_mlf_alpha(sample)

    results["D_MLF"] = np.clip(results["D_MLF"], 0, 0.01)
    D, alpha = results["D_MLF"], results["Alpha"]
    entropy = compute_entropy(sample, D, alpha)
    kurtosis = compute_kurtosis(sample, alpha)
    results["H"], results["K"] = entropy, kurtosis
    save_results(results_dir, results)
    
    fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(15, 10))
    plot_results(results, names=["Alpha", "D_MLF", "K", "H"],
                 fig=fig, axes=np.ravel(axes))
    plt.show()
