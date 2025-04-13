import cv2
import nibabel as nb
import matplotlib.pyplot as plt


img_filename = "../CDMD/sub_005/dwi/sub_005_dwi.nii.gz"
mask_filename = "../CDMD/sub_005/dwi/sub_005_dwi_brainmask.nii.gz"

img = nb.load(img_filename).get_fdata()
mask = nb.load(mask_filename).get_fdata()

axial = img[:, :, 54, 0]
sagittal = img[35, :, :, 0]

axial[mask[:, :, 54] == 0] = 0
sagittal[mask[35, :, :] == 0] = 0

axial = cv2.rotate(axial, cv2.ROTATE_90_COUNTERCLOCKWISE)
sagittal = cv2.rotate(sagittal, cv2.ROTATE_90_COUNTERCLOCKWISE)

fig, axes = plt.subplots(ncols=2, figsize=(20, 10))
axes[0].imshow(axial, cmap="gray")
axes[0].set_title("Axial slice #54", fontsize=18)
axes[0].set_axis_off()
axes[1].imshow(sagittal, cmap="gray")
axes[1].set_title("Sagittal slice #35", fontsize=18)
axes[1].set_axis_off()
fig.tight_layout()
fig.savefig("../fig/slices_masked.png", dpi=300)
plt.show()