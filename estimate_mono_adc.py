import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.CDMD_dataset import CDMDDataset
from src.mono_adc_estimator import estimate_mono_adc
from src.util import save_results, load_results, plot_results,\
    exclude_outliers


if __name__ == "__main__":
    big_delta = 19
    slice_id = 54
    sample_name = "sub_005"
    dataset_root = Path("../CDMD")
    results_dir = Path("../CDMD/results_mono_adc/") / sample_name / str(slice_id)

    dataset = CDMDDataset(dataset_root)
    sample = dataset.load_sample(sample_name, big_delta, slice_id)
    results = load_results(results_dir)

    if not ("ADCmN" in results):
        results = estimate_mono_adc(sample)

    results["ADCmN_raw"] = results["ADCmN"].copy()
    results["ADCmN"] = np.clip(results["ADCmN"], 0, 0.003)
    save_results(results_dir, results)
    
    fig, ax = plot_results(results, names=["ADCmN", "error"])
    plt.show()
    fig.savefig(results_dir / "vis.png", dpi=300)
