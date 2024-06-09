def get_all():
    from .pixelwise import Pixelwise
    from .psnr import PSNR
    from .mse import MSE
    from .ssim import SSIM
    from .ms_ssim import MS_SSIM
    return [Pixelwise(), PSNR(),  SSIM(), MS_SSIM(), MSE()]
