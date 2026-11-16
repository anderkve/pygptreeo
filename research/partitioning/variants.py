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
        elif v == 'None':
            v = None
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
def _oblique(d, seed, dir='pca', pos='median', scaled=False, minfrac=0.25, whiten=False, metric='raw', **opts):
    from oblique import oblique_tree
    kw = dict(BASE)
    kw.update(opts)
    return oblique_tree(direction=dir, position=pos, scaled=scaled, min_fraction=minfrac, whiten=whiten, metric=metric, **kw)


@register('mr')
def _mr(d, seed, min_new=0, **opts):
    from multires import multires_tree
    kw = dict(BASE)
    kw.update(opts)
    return multires_tree(min_new=min_new, **kw)


@register('knn')
def _knn(d, seed, k=100, metric='ard', far=0, search='brute', max_leaves=None, fallback=None,
         dir=None, pos='median', whiten=False, **opts):
    from knn import KNNNode, knn_tree
    kw = dict(BASE)
    kw.update(opts)
    bases, attrs = (KNNNode,), {}
    if dir is not None:                 # oblique cuts under the neighbour conditioning
        from oblique import ObliqueNode
        bases, attrs = (KNNNode, ObliqueNode), dict(direction=dir, position=pos, whiten=whiten)
    return knn_tree(k=k, metric=metric, far=far, search=search, max_leaves=max_leaves, fallback=fallback,
                    node_bases=bases, node_attrs=attrs, **kw)


@register('fisheye')
def _fisheye(d, seed, k=100, per=4, res=32, levels=None, metric='std', search='brute', max_leaves=None, fallback=None,
             **opts):
    from knn import fisheye_tree
    kw = dict(BASE)
    kw.update(opts)
    return fisheye_tree(k=k, per_level=per, reservoir_size=res, levels=levels, metric=metric, search=search,
                        max_leaves=max_leaves, fallback=fallback, **kw)


@register('nb')
def _nb(d, seed, dir=None, pos='median', whiten=False, **opts):
    """The package's own neighbour conditioning, optionally with oblique cuts."""
    kw = dict(BASE)
    kw.update(opts)
    gpt = GPTree(predict_from='neighbours', **kw)
    if dir is not None:
        from oblique import ObliqueNode
        gpt.root.__class__ = type('ObliqueNode_' + dir, (ObliqueNode,), dict(direction=dir, position=pos, whiten=whiten))
    return gpt


@register('nbf')
def _nbf(d, seed, **opts):
    """The package's neighbour conditioning as it ships (the records of 'nb' are of earlier versions)."""
    kw = dict(BASE)
    kw.update(opts)
    return GPTree(predict_from='neighbours', **kw)


@register('anchored')
def _anchored(d, seed, depth=2, **opts):
    """Neighbours of one of 2**depth cells of the query's leaf, shared by the queries of the cell."""
    from anchored import anchored_tree
    kw = dict(BASE)
    kw.update(opts)
    return anchored_tree(depth=depth, **kw)


@register('tapered')
def _tapered(d, seed, start=0.8, strength=1.0, **opts):
    """Neighbours whose noise grows towards the edge of their set, for a prediction that is continuous in the query."""
    from tapered import tapered_tree
    kw = dict(BASE)
    kw.update(opts)
    return tapered_tree(start=start, strength=strength, **kw)


@register('hybrid')
def _hybrid(d, seed, nb=False, **opts):
    """GP leaves on the residual of the feature network, predicting from the leaf or from neighbours."""
    kw = dict(BASE)
    kw.update(opts)
    return GPTree(global_mean='net', global_mean_kwargs=dict(steps_per_update=1, random_state=seed),
                  predict_from='neighbours' if nb else 'leaf', **kw)
