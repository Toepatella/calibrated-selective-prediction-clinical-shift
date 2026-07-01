"""Group-id fold-disjointness discipline (``method_note §1.7``).

From Stage A onward every gate that constructs a split asserts, at the
GROUP-ID level (slide / patient on real data; synthetic group ids on synthetic
data), that its folds -- ``D_cal``, ``D_bbse^src``, ``D_disc``, ``D_tar``,
``D_tar^lab``, ``O``, held-out test -- are mutually disjoint, and emits a printed
"set-intersection = 0" artifact over every fold pair (``method_note §1.7``;
``prereg §2.3``). This is a HARD GATE wherever a gate builds a split
(``build_gates.md §3`` X-CONVENTIONS).
"""

from itertools import combinations

__all__ = ["assert_group_disjoint"]


def assert_group_disjoint(verbose=True, **folds):
    """Assert every pair of the named group-id folds is disjoint.

    Parameters
    ----------
    verbose : bool
        If True, print the ``set-intersection(A, B) = 0`` artifact for every fold pair
        (``method_note §1.7``).
    **folds : name -> iterable of group ids
        Two or more named folds. Group ids may be any hashable (ints, strings,
        patient/slide ids).

    Returns
    -------
    bool
        ``True`` when all pairs are disjoint.

    Raises
    ------
    AssertionError
        If any pair of folds shares one or more group ids.
    """
    sets = {name: set(ids) for name, ids in folds.items()}
    if len(sets) < 2:
        raise ValueError("assert_group_disjoint needs at least two named folds")
    for (na, sa), (nb, sb) in combinations(sets.items(), 2):
        inter = sa & sb
        if verbose:
            # ASCII-only artifact: the printed "set-intersection = 0" evidence must
            # not depend on a UTF-8 console (cp1252 stdout would raise on a glyph).
            print(f"[fold-disjointness] set-intersection({na}, {nb}) = {len(inter)}")
        assert not inter, (
            f"folds {na!r} and {nb!r} share {len(inter)} group id(s): "
            f"{sorted(inter)[:10]}{' ...' if len(inter) > 10 else ''}"
        )
    return True
