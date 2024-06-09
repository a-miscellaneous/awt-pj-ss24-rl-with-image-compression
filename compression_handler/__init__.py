def get_all():
    from .jpeg import JPEG
    # from .jpeg2000 import JPEG2000
    from .webp6 import WEBP6
    from .webp0 import WEBP0
    from .mini_batch_k_means import KMeans
    from .pca import cPCA
    return [JPEG(),  WEBP6(), WEBP0(), KMeans(), cPCA()]
    # return [JPEG(), JPEG2000(), WEBP6(), WEBP0()]
