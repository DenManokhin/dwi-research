from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from src.CDMD_dataset import CDMDDataset
from src.util import load_results


big_delta = 19
slice_id = 54
sample_name = "sub_005"
dataset_root = Path("../CDMD")
results_dir = Path("../CDMD/results_mlf_alpha/") / sample_name / str(slice_id)

dataset = CDMDDataset(dataset_root)
sample = dataset.load_sample(sample_name, big_delta, slice_id)
results = load_results(results_dir)
D = results["H"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_axis_off()
divider = make_axes_locatable(ax)
ax_hist = divider.append_axes("right", size="70%", pad=0.1)

im = ax.imshow(D, cmap="gray")
ax_hist.hist(D[sample.mask != 0], bins=100)
ax_hist.set_yscale("log")
ax_hist.grid(True)

fig.tight_layout()
fig.savefig("../fig/H_hist.png", dpi=300)
plt.show()