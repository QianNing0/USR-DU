import numpy as np


from PerceptualSimilarity.util import util as util_LPIPS
from PerceptualSimilarity.models.util import PerceptualLoss

from code.metrics.metric_util import reorder_image
from code.utils.registry import METRIC_REGISTRY

cri_fea_lpips = PerceptualLoss(model='net-lin', net='alex', use_gpu=False)

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC'):
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are ' '"HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]

    img, img2 = img[:, :, [2, 1, 0]], img2[:, :, [2, 1, 0]]

    img, img2 = util_LPIPS.im2tensor(img), util_LPIPS.im2tensor(img2)

    LPIPS = cri_fea_lpips(img, img2)[0][0][0][0]

    return LPIPS