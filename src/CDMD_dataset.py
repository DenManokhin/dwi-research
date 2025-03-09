import numpy as np
import nibabel as nb
from pathlib import Path
from dataclasses import dataclass


@dataclass
class Sample:
    image: np.ndarray
    timepoints: np.ndarray
    mask: np.ndarray
    delta: float


class CDMDDataset:
    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.Gs = [31, 68, 105, 142, 179, 216, 253, 290]
        self.small_delta = 8
    
    def load_sample(self, sample: str, big_delta: float, slice_id: int) -> Sample:
        sample_subfolder = self.data_root / sample / "dwi" / f"{sample}_dwi"

        img_filename = f"{sample_subfolder}.nii.gz"
        mask_filename = f"{sample_subfolder}_brainmask.nii.gz"
        bvals = np.loadtxt(f"{sample_subfolder}.bval")
        deltas = np.loadtxt(f"{sample_subfolder}.delta")

        delta_mask = ((deltas == 0) | (deltas == big_delta))
        bvals = bvals[delta_mask]
        img = nb.load(img_filename).get_fdata()[:, :, slice_id, delta_mask]
        mask = nb.load(mask_filename).get_fdata()[:, :, slice_id]

        data = []
        for bval in np.unique(bvals):
            bval_mask = (bvals == bval)
            mean_img = np.mean(img[:, :, bval_mask], axis=2)
            data.append({"img": mean_img, "bval": bval})
        data = sorted(data, key=lambda x: x["bval"])

        image = np.dstack([x["img"] for x in data])
        image = np.expand_dims(image, axis=2)
        timepoints = np.asarray([x["bval"] for x in data])
        delta = big_delta - self.small_delta / 3

        sample = Sample(
            image = image,
            timepoints = timepoints,
            mask = mask,
            delta = delta
        )

        return sample
