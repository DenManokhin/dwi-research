from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def save_results(results_dir: Path, results: dict):
    results_dir.mkdir(parents=True, exist_ok=True)
    for key in results:
        np.save(results_dir / f"{key}.npy", results[key])


def load_results(results_dir: Path) -> dict:
    results = {}
    for path in results_dir.glob("*.npy"):
        result = np.load(path)
        results[path.stem] = result
    return results


def plot_results(results: dict, names: list = [], fig=None, axes=None):
    names = names if len(names) > 0 else list(results.keys())
    if fig is None or axes is None:
        fig, axes = plt.subplots(ncols=len(names), figsize=(5 * len(names), 5))
    for ax, name in zip(axes, names):
        result = results[name]
        im = ax.imshow(result, cmap="gray")
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        ax.set_title(name)
    fig.tight_layout()
    return fig, axes


def exclude_outliers(value: np.ndarray, min_quantile: float, max_quantile: float) -> np.ndarray:
    return np.clip(value, *np.quantile(value, [min_quantile, max_quantile]))
