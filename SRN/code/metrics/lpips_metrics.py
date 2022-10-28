import numpy as np

import lpips
from code.metrics.metric_util import reorder_image
from code.utils.registry import METRIC_REGISTRY

cri_fea_lpips = lpips.LPIPS(net='alex')

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

    img, img2 = lpips.im2tensor(img), lpips.im2tensor(img2)

    LPIPS = cri_fea_lpips(img, img2)[0][0][0][0]

    return LPIPS