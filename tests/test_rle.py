from __future__ import annotations

import numpy as np

from backend.app.rle import compact_rle_to_mask, mask_to_compact_rle


def test_compact_rle_round_trip():
    mask = np.zeros((7, 9), dtype=bool)
    mask[2:5, 3:8] = True

    rle = mask_to_compact_rle(mask)
    decoded = compact_rle_to_mask(rle)

    assert isinstance(rle["counts"], str)
    assert rle["size"] == [7, 9]
    assert np.array_equal(decoded, mask)
