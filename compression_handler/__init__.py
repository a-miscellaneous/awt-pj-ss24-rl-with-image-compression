

def ms2020_cc10_mse():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("ms2020-cc10-mse")
    r.parameter_range = range(1, 11)
    r.__class__.__name__ = "ms2020_cc10_mse"
    return r


def ms2020_cc10_mseim():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("ms2020-cc8-msssim")
    r.parameter_range = range(1, 10)
    r.__class__.__name__ = "ms2020_cc10_mseim"
    return r


def mbt2018_mean_mse():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("mbt2018-mean-mse")
    r.parameter_range = range(1, 9)
    r.__class__.__name__ = "mbt2018_mean_mse"
    return r


def mbt2018_mean_msssim():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("mbt2018-mean-msssim")
    r.parameter_range = range(1, 9)
    r.__class__.__name__ = "mbt2018_mean_msssim"
    return r


def bmshj2018_factorized_mse():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("bmshj2018-factorized-mse")
    r.parameter_range = range(1, 9)
    r.__class__.__name__ = "bmshj2018_factorized_mse"
    return r


def bmshj2018_factorized_msssim():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("bmshj2018-factorized-msssim")
    r.parameter_range = range(1, 9)
    r.__class__.__name__ = "bmshj2018_factorized_msssim"
    return r


def bmshj2018_hyperprior_mse():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("bmshj2018-hyperprior-mse")
    r.parameter_range = range(1, 9)
    r.__class__.__name__ = "bmshj2018_hyperprior_mse"
    return r


def bmshj2018_hyperprior_msssim():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("bmshj2018-hyperprior-msssim")
    r.parameter_range = range(1, 9)
    r.__class__.__name__ = "bmshj2018_hyperprior_msssim"
    return r


def b2018_leaky_relu_128():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("b2018-leaky_relu-128")
    r.parameter_range = range(1, 4)
    r.__class__.__name__ = "b2018_leaky_relu_128"
    return r


def b2018_leaky_relu_192():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("b2018-leaky_relu-192")
    r.parameter_range = range(1, 4)
    r.__class__.__name__ = "b2018_leaky_relu_192"
    return r


def b2018_gdn_128():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("b2018-gdn-128")
    r.parameter_range = range(1, 4)
    r.__class__.__name__ = "b2018_gdn_128"
    return r


def b2018_gdn_192():  # works
    from .hific import TFCWrapper
    r = TFCWrapper("b2018-gdn-192")
    r.parameter_range = range(1, 4)
    r.__class__.__name__ = "b2018_gdn_192"
    return r


def get_all():
    from .jpeg import JPEG
    from .jpeg2000 import JPEG2000
    from .webp6 import WEBP6
    from .webp0 import WEBP0
    from .mini_batch_k_means import KMeans
    from .pca import PCA
    return [JPEG(),                             # 0
            WEBP6(),                            # 1
            WEBP0(),                            # 2
            KMeans(),                           # 3
            PCA(),                              # 4
            JPEG2000(),                         # 5
            ms2020_cc10_mse(),                  # 6
            ms2020_cc10_mseim(),                # 7
            mbt2018_mean_mse(),                 # 8
            mbt2018_mean_msssim(),              # 9
            bmshj2018_factorized_mse(),         # 10
            bmshj2018_factorized_msssim(),      # 11
            bmshj2018_hyperprior_mse(),         # 12
            bmshj2018_hyperprior_msssim(),      # 13
            b2018_leaky_relu_128(),             # 14
            b2018_leaky_relu_192(),             # 15
            b2018_gdn_128(),                    # 16
            b2018_gdn_192()                     # 17
            ]

def get_jpeg_comp():
    from .custom_model import customModel
    from .jpeg import JPEG
    return [customModel(), JPEG()]

def get_custom_model():
    from .custom_model import customModel
    return [customModel()]