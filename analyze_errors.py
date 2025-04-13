from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.CDMD_dataset import CDMDDataset
from src.util import load_results


big_delta = 19
slice_id = 54
sample_name = "sub_005"
dataset_root = Path("../CDMD")
results_mlf_dir = Path("../CDMD/results_mlf_alpha/") / sample_name / str(slice_id)
results_mono_dir = Path("../CDMD/results_mono_adc/") / sample_name / str(slice_id)

dataset = CDMDDataset(dataset_root)
sample = dataset.load_sample(sample_name, big_delta, slice_id)
results_mlf = load_results(results_mlf_dir)
results_mono = load_results(results_mono_dir)

error_mlf = results_mlf["error"][sample.mask != 0]
error_mono = results_mono["error"][sample.mask != 0]

quantile = 0.99
quantile_mlf = np.quantile(error_mlf, quantile)
quantile_mono = np.quantile(error_mono, quantile)
error_mlf = error_mlf[error_mlf < quantile_mlf]
error_mono = error_mono[error_mono < quantile_mono]

print(f"MLF errors: mean = {error_mlf.mean()}, quantile{int(quantile*100)} = {quantile_mlf}")
print(f"Mono errors: mean = {error_mono.mean()}, quantile{int(quantile*100)} = {quantile_mono}")

fig, ax = plt.subplots()
ax.hist(error_mlf, bins=100, label="MLF", histtype="step")
ax.hist(error_mono, bins=100, label="Mono", histtype="step")
ax.grid(True)
ax.legend()
ax.set_xlabel("RMSE")
fig.tight_layout()
fig.savefig("../fig/error_hists.png", dpi=300)
plt.show()
