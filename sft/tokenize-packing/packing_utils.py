"""Greedy first-fit packing into fixed-length samples.

Modeled on ``VeOmni/veomni/data/dynamic_batching.py::DynBszBuffer`` but
adapted to fixed-length packs (every emitted pack has length
``max_pack_length``; only the tail of an under-filled pack carries padding).

Output shape (per pack)::

    input_ids  : List[int],   length == max_pack_length
    labels     : List[int],   length == max_pack_length   (padding -> -100)
    cu_seqlens : List[int],   first element 0, last element == max_pack_length
    attn_mask  : List[int],   length == max_pack_length   (1 real / 0 padding)

``cu_seqlens`` ends at ``max_pack_length``: if the pack is fully filled the
last segment is the last real sequence; if the pack has trailing padding the
padding is encoded as a final segment (``cu_seqlens[-2] < cu_seqlens[-1]``).

Packing is per-process: each worker should keep its own ``GreedyPacker``. No
cross-process synchronization is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional


@dataclass
class _BufItem:
    input_ids: List[int]
    labels: List[int]
    length: int
    consumed: bool = False


class GreedyPacker:
    """Buffer-based greedy first-fit packer.

    Parameters
    ----------
    max_pack_length:
        Target token count per emitted pack.
    pad_token_id:
        Pad token id used to fill the tail when a pack is under-filled.
    buffer_size:
        Minimum number of unused items required before ``emit_ready`` produces
        packs during streaming. A larger buffer gives the first-fit walk more
        candidates and improves packing density. ``flush()`` ignores this
        threshold.
    """

    def __init__(self, max_pack_length: int, pad_token_id: int, buffer_size: int = 200) -> None:
        if max_pack_length <= 0:
            raise ValueError("max_pack_length must be positive")
        if buffer_size <= 0:
            raise ValueError("buffer_size must be positive")
        self.max_pack_length = max_pack_length
        self.pad_token_id = pad_token_id
        self.buffer_size = buffer_size
        self._buf: List[_BufItem] = []
        self._active = 0  # cached count of items with consumed=False

    def append(self, input_ids: List[int], labels: List[int]) -> None:
        n = len(input_ids)
        if n != len(labels):
            raise ValueError(f"input_ids/labels length mismatch: {n} vs {len(labels)}")
        if n == 0:
            return
        if n > self.max_pack_length:
            # Oversize samples must be filtered upstream (stage 1 max_doc_length).
            raise ValueError(
                f"sample length {n} > max_pack_length {self.max_pack_length}; "
                "filter at stage 1 with --max-doc-length"
            )
        self._buf.append(_BufItem(list(input_ids), list(labels), n))
        self._active += 1

    def _build_one_pack(self) -> Optional[dict]:
        """First-fit greedy: take the first unused item, then walk and append
        any subsequent unused item that fits the remaining budget.

        Items skipped because they didn't fit remain in the buffer for the next
        pack.
        """
        if self._active == 0:
            return None

        chosen: List[int] = []
        used = 0
        for idx, item in enumerate(self._buf):
            if item.consumed:
                continue
            if not chosen:
                # First fit is always taken (length <= max_pack_length by append's check).
                chosen.append(idx)
                used += item.length
                if used == self.max_pack_length:
                    break
                continue
            if item.length <= self.max_pack_length - used:
                chosen.append(idx)
                used += item.length
                if used == self.max_pack_length:
                    break

        if not chosen:
            return None

        items = []
        for idx in chosen:
            self._buf[idx].consumed = True
            items.append(self._buf[idx])
        self._active -= len(chosen)
        return self._materialize(items)

    def _materialize(self, items: List[_BufItem]) -> dict:
        input_ids: List[int] = []
        labels: List[int] = []
        cu_seqlens: List[int] = [0]
        for item in items:
            input_ids.extend(item.input_ids)
            labels.extend(item.labels)
            cu_seqlens.append(cu_seqlens[-1] + item.length)

        used = cu_seqlens[-1]
        pad = self.max_pack_length - used
        if pad > 0:
            input_ids.extend([self.pad_token_id] * pad)
            labels.extend([-100] * pad)
            cu_seqlens.append(self.max_pack_length)
        # else: cu_seqlens[-1] already == max_pack_length, no trailing pad segment.

        attn_mask = [1] * used + [0] * pad

        assert len(input_ids) == self.max_pack_length
        assert len(labels) == self.max_pack_length
        assert len(attn_mask) == self.max_pack_length
        assert cu_seqlens[0] == 0 and cu_seqlens[-1] == self.max_pack_length

        return {
            "input_ids": input_ids,
            "labels": labels,
            "cu_seqlens": cu_seqlens,
            "attn_mask": attn_mask,
        }

    def _compact(self) -> None:
        if self._active == len(self._buf):
            return
        self._buf = [it for it in self._buf if not it.consumed]

    def emit_ready(self) -> Iterator[dict]:
        """Emit packs while the unused buffer is large enough for good packing.

        Stops once the unused count drops below ``buffer_size`` (or no item
        fits as a first item, which cannot happen with the current append
        check).
        """
        while self._active >= self.buffer_size:
            pack = self._build_one_pack()
            if pack is None:
                break
            yield pack
            self._compact()

    def flush(self) -> Iterator[dict]:
        """Emit remaining packs without the buffer-size threshold. Use at end of stream."""
        while self._active > 0:
            pack = self._build_one_pack()
            if pack is None:
                break
            yield pack
        self._buf = []
        self._active = 0
