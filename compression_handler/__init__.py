def get_all():
    from .jpeg import JPEG
    from .jpeg2000 import JPEG2000
    return [JPEG(), JPEG2000()]
