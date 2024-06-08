def get_all():
    from .pixelwise import Pixelwise
    from .psnr import PSNR
    from .mse import MSE
    from .ssim import SSIM
    return [Pixelwise(), PSNR(), MSE(), SSIM()]
