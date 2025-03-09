import cv2
import numpy as np
from dwilib.fit import Model


def restore_masked_result(result: np.ndarray, mask: np.ndarray) -> np.ndarray:
    restored_results = []
    for i in range(result.shape[-1]):
        img = np.zeros_like(mask, dtype=np.float64)
        img[mask != 0] = result[:, i].flatten()
        restored_results.append(img)
    return np.dstack(restored_results)


def fit(
        image: np.ndarray,
        timepoints: np.ndarray,
        model: Model,
        mask: "np.ndarray|None" = None,
        use_scipy: bool = False,
        multiprocess: bool = False,
        known_params: np.ndarray = None
    ) -> dict:
    """
    Fit model to image.
        image - (W, H, n_slices, n_bvalues) image
        timepoints - b-values
        model - Models
    """
    shape = image.shape[:-1]
    if mask is not None:
        image = image[np.where(mask != 0)]
    image = image.reshape(-1, len(timepoints))
    assert len(timepoints) == len(image[0]), len(image[0])

    pmap = model.fit(timepoints, image, use_scipy, known_params, multiprocess)

    if mask is not None:
        pmap = restore_masked_result(pmap, mask)
    else:
        pmap = pmap.reshape(shape + (pmap.shape[-1],))

    pmap_names = [param.name for param in model.params] + ["error"]
    pmap_values = cv2.split(pmap)
    pmaps_dict = dict(zip(pmap_names, pmap_values))

    return pmaps_dict
