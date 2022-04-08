from lib.models.monodle import MonoDLE
from lib.models.monocon import MonoCon

def build_model(cfg):
    if cfg['type'] == 'monocon':
        return MonoCon(backbone=cfg['backbone'], neck=cfg['neck'], num_class=cfg['num_class'])
    elif cfg['type'] == 'monodle':
        return MonoDLE(backbone=cfg['backbone'], neck=cfg['neck'], num_class=cfg['num_class'])
    else:
        raise NotImplementedError("%s model is not supported" % cfg['type'])


