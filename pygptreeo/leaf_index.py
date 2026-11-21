"""A balanced index over the leaves of a tree.

The tree of cuts defines the cells of the leaves, but its depth follows the
order in which the points arrive: on a stream that drifts, every split adds a
level. ``LeafIndex`` keeps the same leaves in a hierarchy of bounding boxes
whose depth grows with the logarithm of their number whatever the order. It
holds three boxes per leaf: the cell its cuts give it, used to find the leaf
of a point; the bounding box of its stored points, used to find the leaves
that can hold a point's nearest neighbours; and its reach, the box of its
faces widened by a zone at each cut, used to find the leaves that take part
in a prediction.
"""

import heapq

import numpy as np


ARRAYS = ('cell_lo', 'cell_hi', 'data_lo', 'data_hi', 'face_lo', 'face_hi', 'half', 'reach_lo', 'reach_hi')


class _Group:
    """A node of the index: up to ``fanout`` leaves or groups, with their boxes
    in the rows of the arrays named in ``ARRAYS``. The faces and the half
    widths of the zones are those of leaves; a row of a group above the
    leaves holds zeros there."""

    __slots__ = ('parent', 'children', 'bottom') + ARRAYS

    def __init__(self, bottom, capacity, n_features):
        self.parent = None
        self.children = []
        self.bottom = bottom            # True if the children are leaves
        for name in ARRAYS:
            setattr(self, name, np.empty((capacity, n_features)))

    def boxes(self, i):
        return tuple(getattr(self, name)[i] for name in ARRAYS)

    def put(self, i, child, boxes):
        if i == len(self.children):
            self.children.append(child)
        else:
            self.children[i] = child
        for name, box in zip(ARRAYS, boxes):
            getattr(self, name)[i] = box
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
        nothing = np.zeros(self.cell_lo.shape[1])
        return (self.cell_lo[:n].min(axis=0), self.cell_hi[:n].max(axis=0),
                self.data_lo[:n].min(axis=0), self.data_hi[:n].max(axis=0), nothing, nothing, nothing,
                self.reach_lo[:n].min(axis=0), self.reach_hi[:n].max(axis=0))


class LeafIndex:
    """A hierarchy of bounding boxes over the leaves of a tree.

    A cell is the half-open box ``lo <= x < hi`` (infinite where no cut bounds
    it) of the points that belong to the leaf alone: it stops short of a cut
    by half the overlap of the two children. The box of a leaf's points is
    empty (``lo = inf``, ``hi = -inf``) while the leaf holds none. The faces
    of a leaf are its cuts themselves. With ``blend`` a leaf has a zone at
    each face, half of it on either side, as wide as the share ``blend`` of
    the leaf's width along that input (to the edge of its points where no
    cut bounds it); its reach is the box of its faces with the outer halves
    of its zones. The boxes of a group contain those of its children, and
    are not shrunk when a leaf is removed.
    """

    def __init__(self, n_features: int, fanout: int = 32, blend=None):
        self.n_features = n_features
        self.fanout = fanout
        self.blend_width = 0.0 if blend is None else blend
        self.root = _Group(True, fanout + 1, n_features)
        self._group = {}                # leaf -> the group that holds it

    def __len__(self):
        return len(self._group)

    @staticmethod
    def data_box(leaf):
        """The bounding box of the points a leaf stores."""
        X = leaf.my_X_data
        if X.shape[0] == 0:
            d = leaf.n_features
            return np.full(d, np.inf), np.full(d, -np.inf)
        return X.min(axis=0), X.max(axis=0)

    def add(self, leaf):
        """Add the first leaf of a tree; its cell is the whole space."""
        d = self.n_features
        everything = (np.full(d, -np.inf), np.full(d, np.inf))
        self.root.put(len(self.root.children), leaf,
                      everything + self.data_box(leaf) + everything + (np.zeros(d),) + everything)
        self._group[leaf] = self.root

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
        face_lo, face_hi = group.face_lo[i].copy(), group.face_hi[i].copy()
        left_face_hi, right_face_lo = face_hi.copy(), face_lo.copy()
        left_face_hi[j] = right_face_lo[j] = position
        # the extent of the leaf: to its faces, or to the edge of its points where it has none
        with_points = group.data_lo[i, 0] <= group.data_hi[i, 0]
        extent_lo = np.where(np.isfinite(face_lo), face_lo, group.data_lo[i] if with_points else np.nan)
        extent_hi = np.where(np.isfinite(face_hi), face_hi, group.data_hi[i] if with_points else np.nan)
        left_width, right_width = extent_hi - extent_lo, extent_hi - extent_lo
        left_width[j], right_width[j] = position - extent_lo[j], extent_hi[j] - position
        left_width, right_width = (np.nan_to_num(np.maximum(width, 0.0)) for width in (left_width, right_width))
        # a child without a width along the cut has its sibling's zone there, for weights without a step
        if left_width[j] == 0.0:
            left_width[j] = right_width[j]
        elif right_width[j] == 0.0:
            right_width[j] = left_width[j]
        left_half, right_half = 0.5 * self.blend_width * left_width, 0.5 * self.blend_width * right_width
        group.put(i, left, (lo, left_hi) + self.data_box(left)
                  + (face_lo, left_face_hi, left_half, face_lo - left_half, left_face_hi + left_half))
        group.put(len(group.children), right, (right_lo, hi) + self.data_box(right)
                  + (right_face_lo, face_hi, right_half, right_face_lo - right_half, face_hi + right_half))
        self._group[left] = self._group[right] = group
        self._grow(group)
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

    def release(self):
        """Break the links from the groups to their parents, so that an index
        that is dropped is freed without the garbage collector."""
        groups = [self.root]
        for group in groups:            # the list grows while it is read
            if not group.bottom:
                groups += group.children
            group.parent = None

    def extend(self, leaf, removed, j, lower):
        """Extend a leaf along input ``j`` over a leaf beside it that is about
        to be removed, on its ``lower`` side or the other: its cell and its
        face there become those of the removed leaf, and its width the sum of
        the two."""
        gone = self._group[removed]
        k = gone.children.index(removed)
        group = self._group[leaf]
        i = group.children.index(leaf)
        if lower:
            group.cell_lo[i, j], group.face_lo[i, j] = gone.cell_lo[k, j], gone.face_lo[k, j]
        else:
            group.cell_hi[i, j], group.face_hi[i, j] = gone.cell_hi[k, j], gone.face_hi[k, j]
        group.half[i, j] += gone.half[k, j]
        group.reach_lo[i] = group.face_lo[i] - group.half[i]
        group.reach_hi[i] = group.face_hi[i] + group.half[i]
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

    def blend(self, q):
        """The leaves that take part in a prediction at ``q`` and their weights,
        which sum to 1: ``(leaves, weights)``. Before the weights are
        normalised, that of a leaf is the product over the inputs of a
        factor that is 1 within its faces less half a zone, falls linearly
        to a half at a face and to 0 half a zone beyond it."""
        leaves, weights = [], []
        stack = [self.root]
        while stack:
            group = stack.pop()
            n = len(group.children)
            inside = np.flatnonzero(((group.reach_lo[:n] <= q) & (q < group.reach_hi[:n])).all(axis=1))
            if not group.bottom:
                stack.extend([group.children[i] for i in inside])
            elif inside.size:
                zone = 2.0 * group.half[inside]
                with np.errstate(divide='ignore', invalid='ignore'):
                    rise = np.minimum((q - group.reach_lo[inside]) / zone, (group.reach_hi[inside] - q) / zone)
                weight = np.clip(np.where(zone > 0.0, rise, 1.0), 0.0, 1.0).prod(axis=1)
                for i, value in zip(inside, weight):
                    if value > 0.0:
                        leaves.append(group.children[i])
                        weights.append(value)
        total = sum(weights)
        return leaves, [weight / total for weight in weights]

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
