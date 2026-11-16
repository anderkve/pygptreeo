"""A multi-resolution tree: a node that splits leaves its GP behind as a frozen
coarse model, and the GPs of its descendants model what is left of the target."""
import numpy as np

from pygptreeo import GPTree
from pygptreeo.gpnode import GPNode


class FrozenMean:
    """The summed GP means of a split node and of its ancestors."""

    def __init__(self, parent, node):
        self.parent = parent
        self.version = 1 + (parent.version if parent is not None else 0)
        self.x_mean, self.x_scale = node.X_scaler.mean_.copy(), node.X_scaler.scale_.copy()
        self.parts = []
        for gpr, ys in zip(node.my_GPRs, node.y_scalers):
            sk = gpr.sklearn_gpr
            self.parts.append((sk.kernel_, sk.X_train_, sk.alpha_.reshape(-1), ys.mean_[0], ys.scale_[0]))

    @property
    def depth(self):
        return self.version

    def predict(self, X):
        X = np.atleast_2d(X)
        Xs = (X - self.x_mean) / self.x_scale
        out = np.column_stack([kernel(Xs, Xtr) @ alpha * scale + shift
                               for kernel, Xtr, alpha, shift, scale in self.parts])
        if self.parent is not None:
            out = out + self.parent.predict(X)
        return out


class _Holder:
    def __init__(self, current):
        self.current = current


class MRNode(GPNode):
    """A node whose GPs model the residual of its ancestors' frozen means (``chain``).

    Class attributes:
        min_new: a child takes up its parent's frozen mean at its first fit with
            at least this many points of its own that the parent had not seen
    """
    chain = None
    pending = None
    n_inherited = 0
    min_new = 0

    def fit_my_GPR(self, force_training=False):
        if self.pending is not None and self.n_points - self.n_inherited >= self.min_new:
            self.chain, self.pending = self.pending, None
        self.global_mean = _Holder(self.chain) if self.chain is not None else None
        try:
            return super().fit_my_GPR(force_training=force_training)
        finally:
            self.global_mean = None

    def generate_children(self, GPR, n_features):
        super().generate_children(GPR, n_features)
        frozen = self._fitted_global
        if self._is_fitted() and self.output_model == 'independent' and self.X_scaler is not None:
            frozen = FrozenMean(self._fitted_global, self)
        for child in self.children:
            child.chain = self._fitted_global
            child.pending = frozen

    def split_training_data(self):
        super().split_training_data()
        for child in self.children:
            child.n_inherited = child.n_points


def multires_tree(min_new=0, **tree_kwargs):
    node_class = type('MRNode_', (MRNode,), dict(min_new=min_new))
    gpt = GPTree(**tree_kwargs)
    gpt.root.__class__ = node_class
    return gpt
