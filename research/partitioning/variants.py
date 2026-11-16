"""Registry of the model variants compared in the partitioning study.

``make_model(name, d, seed)`` returns a fresh model. A variant name may carry
options after a colon, e.g. ``oblique:dir=pca``.
"""
from pygptreeo import GPTree

BASE = dict(Nbar=100, theta=1e-4, retrain_every_n_points=25, splitting_strategy='gradual')

REGISTRY = {}


def register(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def parse(name):
    """'kind:a=1,b=x' -> ('kind', {'a': 1, 'b': 'x'})"""
    kind, _, rest = name.partition(':')
    opts = {}
    for item in filter(None, rest.split(',')):
        k, _, v = item.partition('=')
        for cast in (int, float):
            try:
                v = cast(v)
                break
            except ValueError:
                pass
        if v == 'True' or v == 'False':
            v = (v == 'True')
        opts[k] = v
    return kind, opts


def make_model(name, d, seed):
    kind, opts = parse(name)
    return REGISTRY[kind](d=d, seed=seed, **opts)


@register('base')
def _base(d, seed, **opts):
    kw = dict(BASE)
    kw.update(opts)
    return GPTree(**kw)


@register('oblique')
def _oblique(d, seed, dir='pca', pos='median', scaled=False, minfrac=0.25, **opts):
    from oblique import oblique_tree
    kw = dict(BASE)
    kw.update(opts)
    return oblique_tree(direction=dir, position=pos, scaled=scaled, min_fraction=minfrac, **kw)


@register('mr')
def _mr(d, seed, min_new=0, **opts):
    from multires import multires_tree
    kw = dict(BASE)
    kw.update(opts)
    return multires_tree(min_new=min_new, **kw)


@register('knn')
def _knn(d, seed, k=100, metric='ard', far=0, search='brute', max_leaves=None, **opts):
    from knn import knn_tree
    kw = dict(BASE)
    kw.update(opts)
    return knn_tree(k=k, metric=metric, far=far, search=search, max_leaves=max_leaves, **kw)


@register('fisheye')
def _fisheye(d, seed, k=100, per=4, res=32, levels=None, metric='ard', search='brute', max_leaves=None, **opts):
    from knn import fisheye_tree
    kw = dict(BASE)
    kw.update(opts)
    return fisheye_tree(k=k, per_level=per, reservoir_size=res, levels=levels, metric=metric, search=search,
                        max_leaves=max_leaves, **kw)
