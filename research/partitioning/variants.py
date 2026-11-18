"""Registry of the model variants compared in the partitioning study.

``make_model(name, d, seed)`` returns a fresh model. A variant name may carry
options after a colon, e.g. ``oblique:dir=pca``.
"""
from pygptreeo import GPTree

BASE = dict(Nbar=100, theta=1e-4, retrain_every_n_points=25, splitting_strategy='gradual')

# the settings of a tree's neighbour search: its number of neighbours, and the rest by the names of the package's NEIGHBOUR_ constants
SETTINGS = {'n_neighbours': 'n_neighbours', 'neighbour_leaves': 'max_leaves', 'neighbour_fade': 'fade', 'neighbour_blend': 'blend', 'neighbour_awake': 'awake',
            'neighbour_consistency': 'consistency_limit', 'neighbour_consistency_outside': 'outside_limit',
            'neighbour_transition': 'transition'}


def neighbour_tree(**kw):
    """A tree that predicts from neighbours. Options named as in SETTINGS set
    its search in place of the package's constants and of ``Nbar`` neighbours,
    e.g. ``neighbour_fade=None`` or ``n_neighbours=150``."""
    settings = {SETTINGS[name]: kw.pop(name) for name in list(kw) if name in SETTINGS}
    kw.setdefault('predict_from', 'neighbours')
    tree = GPTree(**kw)
    if tree.neighbours is not None:
        for name, value in settings.items():
            setattr(tree.neighbours, name, value)
    return tree

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
    gpt = neighbour_tree(**kw)
    if dir is not None:
        from oblique import ObliqueNode
        gpt.root.__class__ = type('ObliqueNode_' + dir, (ObliqueNode,), dict(direction=dir, position=pos, whiten=whiten))
    return gpt


@register('nbf')
def _nbf(d, seed, **opts):
    """The package's neighbour conditioning as it ships (the records of 'nb' are of earlier versions)."""
    kw = dict(BASE)
    kw.update(opts)
    return neighbour_tree(**kw)


@register('guarded')
def _guarded(d, seed, measure='centroid', far=0.65, limit=3.0, **opts):
    """The shipped neighbours with a stricter test against the kernel for a query outside them."""
    from guarded import guarded_tree
    kw = dict(BASE)
    kw.update(opts)
    return guarded_tree(measure=measure, far=far, limit=limit, **kw)


@register('withleaf')
def _withleaf(d, seed, **opts):
    """The shipped neighbours together with the rest of the leaf's own points."""
    from with_leaf import with_leaf_tree
    kw = dict(BASE)
    kw.update(opts)
    return with_leaf_tree(**kw)


@register('smooth')
def _smooth(d, seed, inside=0.5, outside=0.8, far=3.0, width=2.0, **opts):
    """The shipped neighbours with a gradual change to the leaf's own prediction."""
    from smooth_guard import smooth_tree
    kw = dict(BASE)
    kw.update(opts)
    return smooth_tree(inside=inside, outside=outside, far=far, width=width, **kw)


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


@register('blended')
def _blended(d, seed, blend=0.1, **opts):
    """Neighbours, with the predictions of the leaves on both sides of a cut mixed in a zone around it."""
    from blended import blended_tree
    kw = dict(BASE)
    kw.update(opts)
    return blended_tree(blend=blend, **kw)


@register('mixed')
def _mixed(d, seed, blend=0.1, **opts):
    """As 'blended', with one prediction from the mixture of the kernels of the leaves on both sides of a cut."""
    from blended import mixed_tree
    kw = dict(BASE)
    kw.update(opts)
    return mixed_tree(blend=blend, **kw)


@register('hybrid')
def _hybrid(d, seed, nb=False, **opts):
    """GP leaves on the residual of the feature network, predicting from the leaf or from neighbours."""
    kw = dict(BASE)
    kw.update(opts)
    return neighbour_tree(global_mean='net', global_mean_kwargs=dict(steps_per_update=1, random_state=seed),
                          predict_from='neighbours' if nb else 'leaf', **kw)
