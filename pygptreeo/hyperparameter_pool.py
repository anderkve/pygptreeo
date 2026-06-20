"""Tree-global pooling of GP kernel hyperparameters.

This module implements a lightweight mechanism for sharing learned kernel
hyperparameters across all leaf nodes of a GPTree. The motivation is sample
efficiency: every leaf models the *same* underlying function in a different
region of the input space, so the optimal kernel hyperparameters are usually
similar. A freshly created leaf with only a handful of points cannot reliably
estimate, e.g., its length scales on its own, but it can borrow a robust
consensus estimate pooled from the more mature leaves.

When leaves use standard scaling (the default), their training data lives in a
common standardized coordinate system, which makes the learned hyperparameters
directly comparable and therefore well suited for pooling. This is what allows
hyperparameter pooling to work *together* with standard scaling, unlike the
parent-to-child ``use_hyperparameter_inheritance`` mechanism.

The pool stores hyperparameters in the backend-native log-space representation
(``theta`` for scikit-learn kernels) and aggregates them with an elementwise
median, which is robust to the occasional badly-fit leaf.
"""

import numpy as np


class HyperparameterPool:
    """Stores and aggregates GP kernel hyperparameters across leaf nodes.

    A single ``HyperparameterPool`` instance is created by a ``GPTree`` and
    shared by reference among all of its ``GPNode`` instances. Each leaf
    contributes its latest set of learned hyperparameters after it is trained,
    keyed by the node name so that re-training a leaf overwrites its previous
    contribution rather than adding a duplicate. When a node stops being a leaf
    (i.e., it splits), its contribution is removed.

    The aggregated estimate is the elementwise median of the contributions,
    computed independently for each output dimension.

    Attributes:
        enabled (bool): If False, the pool is inert and callers skip pooling.
        n_outputs (int): Number of output dimensions tracked.
    """

    def __init__(self, n_outputs: int = 1, enabled: bool = False):
        """Initializes the pool.

        Args:
            n_outputs (int): Number of output dimensions. One independent pool
                of hyperparameters is kept per output. Defaults to 1.
            enabled (bool): Whether pooling is active. Defaults to False so that
                the default GPTree behaviour is unchanged.
        """
        self.enabled = enabled
        self.n_outputs = n_outputs
        # One dict per output: {node_name: theta_vector}
        self._thetas = [dict() for _ in range(n_outputs)]

    def update(self, name: str, output_index: int, theta) -> None:
        """Records (or overwrites) a leaf's hyperparameters for one output.

        Args:
            name (str): The contributing node's name.
            output_index (int): Which output dimension the hyperparameters
                belong to.
            theta: The hyperparameter vector (log-space). If None, the call is
                ignored (e.g., for GP backends that do not expose
                hyperparameters).
        """
        if theta is None:
            return
        self._thetas[output_index][name] = np.asarray(theta, dtype=float)

    def remove(self, name: str) -> None:
        """Removes a node's contributions across all outputs.

        Called when a node splits and is no longer a leaf, so that stale
        hyperparameters from an internal node no longer bias the estimate.

        Args:
            name (str): The node name to remove.
        """
        for d in self._thetas:
            d.pop(name, None)

    def estimate(self, output_index: int):
        """Returns the pooled hyperparameter estimate for one output.

        Args:
            output_index (int): Which output dimension to estimate.

        Returns:
            np.ndarray or None: The elementwise median of all current
            contributions, or None if no leaf has contributed yet (or the
            contributions have inconsistent shapes).
        """
        vals = list(self._thetas[output_index].values())
        if not vals:
            return None
        # All leaves clone the same kernel structure, so theta lengths match.
        # Guard defensively against any mismatch.
        lengths = {v.shape[0] for v in vals}
        if len(lengths) != 1:
            return None
        return np.median(np.vstack(vals), axis=0)

    def n_contributors(self, output_index: int = 0) -> int:
        """Returns the number of leaves currently contributing to one output."""
        return len(self._thetas[output_index])
