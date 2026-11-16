"""A balanced index over the leaves of a tree.

The tree of cuts defines the cells of the leaves, but its depth follows the
order in which the points arrive: on a stream that drifts, every split adds a
level. ``LeafIndex`` keeps the same leaves in a hierarchy of bounding boxes
whose depth grows with the logarithm of their number whatever the order. It
holds two boxes per leaf: the cell its cuts give it, used to find the leaf of
a point, and the bounding box of its stored points, used to find the leaves
that can hold a point's nearest neighbours.
"""

import heapq

import numpy as np


class _Group:
    """A node of the index: up to ``fanout`` leaves or groups, with their boxes
    in the rows of four arrays."""

    __slots__ = ('parent', 'children', 'bottom', 'cell_lo', 'cell_hi', 'data_lo', 'data_hi')

    def __init__(self, bottom, capacity, n_features):
        self.parent = None
        self.children = []
        self.bottom = bottom            # True if the children are leaves
        self.cell_lo = np.empty((capacity, n_features))
        self.cell_hi = np.empty((capacity, n_features))
        self.data_lo = np.empty((capacity, n_features))
        self.data_hi = np.empty((capacity, n_features))

    def boxes(self, i):
        return self.cell_lo[i], self.cell_hi[i], self.data_lo[i], self.data_hi[i]

    def put(self, i, child, boxes):
        if i == len(self.children):
            self.children.append(child)
        else:
            self.children[i] = child
        self.cell_lo[i], self.cell_hi[i], self.data_lo[i], self.data_hi[i] = boxes
        if not self.bottom:
            child.parent = self

    def drop(self, i):
        """Remove child ``i``; the last child takes its row."""
        last = len(self.children) - 1
        if i != last:
            self.put(i, self.children[last], [a.copy() for a in self.boxes(last)])
        self.children.pop()

    def union(self):
        n = len(self.children)
        return (self.cell_lo[:n].min(axis=0), self.cell_hi[:n].max(axis=0),
                self.data_lo[:n].min(axis=0), self.data_hi[:n].max(axis=0))


class LeafIndex:
    """A hierarchy of bounding boxes over the leaves of a tree.

    A cell is the half-open box ``lo <= x < hi`` (infinite where no cut bounds
    it) of the points that belong to the leaf alone: it stops short of a cut
    by half the overlap of the two children. The box of a leaf's points is
    empty (``lo = inf``, ``hi = -inf``) while the leaf holds none. The boxes
    of a group contain those of its children; they are not shrunk when a leaf
    is removed.

    Args:
        n_features (int): Number of inputs.
        fanout (int): Largest number of children of a group.
    """

    def __init__(self, n_features: int, fanout: int = 32):
        self.n_features = n_features
        self.fanout = fanout
        self.root = _Group(True, fanout + 1, n_features)
        self._group = {}                # leaf -> the group that holds it

    def __len__(self):
        return len(self._group)

    def __contains__(self, leaf):
        return leaf in self._group

    @staticmethod
    def data_box(leaf):
        """The bounding box of the points a leaf stores."""
        X = leaf.my_X_data
        if X is None or X.shape[0] == 0:
            d = leaf.n_features
            return np.full(d, np.inf), np.full(d, -np.inf)
        return X.min(axis=0), X.max(axis=0)

    def add(self, leaf):
        """Add the first leaf of a tree; its cell is the whole space."""
        d = self.n_features
        self.root.put(len(self.root.children), leaf,
                      (np.full(d, -np.inf), np.full(d, np.inf)) + self.data_box(leaf))
        self._group[leaf] = self.root

    def cell(self, leaf):
        """The cell of a leaf: ``(lo, hi)``."""
        group = self._group[leaf]
        i = group.children.index(leaf)
        return group.cell_lo[i].copy(), group.cell_hi[i].copy()

    def split(self, leaf, left, right, j, position, margin=0.0):
        """Replace a leaf by its two children, cut along input ``j`` at
        ``position``. Their cells stop ``margin`` short of the cut, and a
        point on a cut without a margin is in neither."""
        group = self._group.pop(leaf)
        i = group.children.index(leaf)
        lo, hi = group.cell_lo[i].copy(), group.cell_hi[i].copy()
        left_hi, right_lo = hi.copy(), lo.copy()
        left_hi[j] = min(hi[j], position - margin)
        right_lo[j] = max(lo[j], position + margin if margin > 0 else np.nextafter(position, np.inf))
        group.put(i, left, (lo, left_hi) + self.data_box(left))
        group.put(len(group.children), right, (right_lo, hi) + self.data_box(right))
        self._group[left] = self._group[right] = group
        if len(group.children) > self.fanout:
            self._divide(group)

    def remove(self, leaf):
        """Take a leaf out of the index."""
        group = self._group.pop(leaf)
        group.drop(group.children.index(leaf))
        while not group.children and group.parent is not None:
            parent = group.parent
            parent.drop(parent.children.index(group))
            group = parent

    def set_cell(self, leaf, lo, hi):
        """Give a leaf a new cell, which contains its old one."""
        group = self._group[leaf]
        i = group.children.index(leaf)
        group.cell_lo[i], group.cell_hi[i] = lo, hi
        self._grow(group)

    def grow(self, leaf, x):
        """Extend the box of a leaf's points to the point ``x``."""
        group = self._group[leaf]
        i = group.children.index(leaf)
        lo, hi = group.data_lo[i], group.data_hi[i]
        if (x < lo).any() or (x > hi).any():
            np.minimum(lo, x, out=lo)
            np.maximum(hi, x, out=hi)
            self._grow(group)

    def _grow(self, group):
        """Extend the boxes of the groups above ``group`` to those of their children."""
        while group.parent is not None:
            parent = group.parent
            i = parent.children.index(group)
            union = group.union()
            if all((old == new).all() for old, new in zip(parent.boxes(i), union)):
                break
            parent.put(i, group, union)
            group = parent

    def _divide(self, group):
        """Split a group that holds more than ``fanout`` children into two: along
        the input, and with 30 to 70% of the children in the first half, for
        which the boxes of the two halves overlap least."""
        n = len(group.children)
        order, size = np.arange(n), n // 2
        filled = group.data_lo[:n, 0] <= group.data_hi[:n, 0]
        if filled.any():
            # the cells of the children within the reach of the points
            reach_lo, reach_hi = group.data_lo[:n][filled].min(axis=0), group.data_hi[:n][filled].max(axis=0)
            lo = np.clip(group.cell_lo[:n], reach_lo, reach_hi)
            hi = np.clip(group.cell_hi[:n], reach_lo, reach_hi)
            centre = 0.5 * (lo + hi)
            sizes = np.arange(max(1, -(-3 * n // 10)), min(n - 1, 7 * n // 10) + 1)
            best = None
            for j in range(self.n_features):
                by_centre = np.argsort(centre[:, j], kind='stable')
                upper = np.maximum.accumulate(hi[by_centre, j])                 # of the first children
                lower = np.minimum.accumulate(lo[by_centre[::-1], j])[::-1]     # of the last children
                extent = upper[-1] - lower[0]
                if extent <= 0.0:
                    continue
                overlap = np.maximum(upper[sizes - 1] - lower[sizes], 0.0) / extent
                i = np.lexsort((np.abs(sizes - n / 2), overlap))[0]             # least overlap, then most even
                key = (overlap[i], abs(sizes[i] - n / 2))
                if best is None or key < best:
                    best, order, size = key, by_centre, int(sizes[i])
        members = [(group.children[i], [a.copy() for a in group.boxes(i)]) for i in order]
        other = _Group(group.bottom, self.fanout + 1, self.n_features)
        group.children = []
        for k, (child, boxes) in enumerate(members):
            target = group if k < size else other
            target.put(len(target.children), child, boxes)
            if target.bottom:
                self._group[child] = target
        parent = group.parent
        if parent is None:
            parent = self.root = _Group(False, self.fanout + 1, self.n_features)
            parent.put(0, group, group.union())
        else:
            parent.put(parent.children.index(group), group, group.union())
        parent.put(len(parent.children), other, other.union())
        if len(parent.children) > self.fanout:
            self._divide(parent)

    def locate(self, q):
        """The leaf whose cell contains the point ``q``, or None if no cell does."""
        stack = [self.root]
        while stack:
            group = stack.pop()
            n = len(group.children)
            inside = np.flatnonzero(((group.cell_lo[:n] <= q) & (q < group.cell_hi[:n])).all(axis=1))
            if group.bottom:
                if inside.size:
                    return group.children[inside[0]]
            else:
                stack.extend([group.children[i] for i in inside])
        return None

    def within(self, q, w, radius):
        """The leaves that can hold a point within ``radius`` (a squared distance)
        of ``q``, with every input divided by ``w``: ``(leaves, bounds)``, where a
        bound is the squared distance from ``q`` to the box of a leaf's points."""
        leaves, bounds = [], []
        stack = [self.root]
        while stack:
            group = stack.pop()
            n = len(group.children)
            gap = np.maximum(group.data_lo[:n] - q, 0.0) + np.maximum(q - group.data_hi[:n], 0.0)
            bound = ((gap / w) ** 2).sum(axis=1)
            near = np.flatnonzero(bound <= radius)
            if group.bottom:
                leaves.extend([group.children[i] for i in near])
                bounds.append(bound[near])
            else:
                stack.extend([group.children[i] for i in near])
        return leaves, (np.concatenate(bounds) if bounds else np.empty(0))

    def nearest(self, q, w, m):
        """The ``m`` leaves whose points' boxes are nearest to ``q``, with every
        input divided by ``w``, nearest first. Leaves without points are left out."""
        leaves, bounds = [], []
        limit = np.inf                  # the m-th smallest bound found
        heap = [(0.0, 0, self.root)]
        pushed = 1
        while heap:
            bound, _, group = heapq.heappop(heap)
            if bound >= limit:
                break
            n = len(group.children)
            gap = np.maximum(group.data_lo[:n] - q, 0.0) + np.maximum(q - group.data_hi[:n], 0.0)
            below = ((gap / w) ** 2).sum(axis=1)
            for i in np.flatnonzero(below < limit):
                if group.bottom:
                    leaves.append(group.children[i])
                    bounds.append(below[i])
                else:
                    heapq.heappush(heap, (below[i], pushed, group.children[i]))
                    pushed += 1
            if group.bottom and len(bounds) >= m:
                limit = np.partition(bounds, m - 1)[m - 1]
        order = np.argsort(bounds, kind='stable')[:m]
        return [leaves[i] for i in order]

    def siblings(self, leaf, q, w):
        """The leaves in the group of ``leaf``, nearest to ``q`` first."""
        group = self._group[leaf]
        n = len(group.children)
        gap = np.maximum(group.data_lo[:n] - q, 0.0) + np.maximum(q - group.data_hi[:n], 0.0)
        return [group.children[i] for i in np.argsort(((gap / w) ** 2).sum(axis=1), kind='stable')]

    def depth(self):
        """Number of levels of groups above the leaves."""
        levels, group = 1, self.root
        while not group.bottom:
            group = group.children[0]
            levels += 1
        return levels
