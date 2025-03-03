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
        known_params: np.ndarray = None
    ) -> np.ndarray:
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

    pmap = model.fit(timepoints, image)

    if mask is not None:
        pmap = restore_masked_result(pmap, mask)
    else:
        pmap = pmap.reshape(shape + (pmap.shape[-1],))
    return pmap
