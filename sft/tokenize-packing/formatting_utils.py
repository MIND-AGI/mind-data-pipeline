"""Chat-template tokenization + assistant-only loss masking.

Self-contained (no nemo_automodel imports). Adapted from
sft_sample/mind_sft_formatting_utils.py and sft_sample/mind_sft_dataset.py.

The single public entry point is `tokenize_sample`, which takes a raw chat row
(OpenAI `messages` format, optional `tools`) and returns
``{"input_ids": List[int], "labels": List[int]}``.

`labels` is the next-token-prediction shifted target: positions that are not
assistant tokens are set to -100.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

GENERATION_REGEX = re.compile(r"\{%-?\s+generation\s+-?%\}")


def has_chat_template(tokenizer) -> bool:
    return getattr(tokenizer, "chat_template", None) is not None and callable(
        getattr(tokenizer, "apply_chat_template", None)
    )


def ensure_pad_token(tokenizer) -> int:
    if getattr(tokenizer, "pad_token_id", None) is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer.pad_token_id


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Coerce content to strings; parse tool_call argument JSON strings into dicts."""
    norm: List[Dict[str, Any]] = []
    for m in messages:
        role = m.get("role")
        if role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"Unsupported role in messages: {role}")
        out = dict(m)
        content = m.get("content")
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            out["content"] = "".join(text_parts) if text_parts else ""
        elif content is None:
            out["content"] = ""
        else:
            out["content"] = str(content)

        tool_calls = out.get("tool_calls")
        if isinstance(tool_calls, list):
            new_calls = []
            for call in tool_calls:
                if not isinstance(call, dict):
                    new_calls.append(call)
                    continue
                call_copy = dict(call)
                fn = call_copy.get("function")
                if isinstance(fn, dict):
                    fn_copy = dict(fn)
                    args = fn_copy.get("arguments")
                    if isinstance(args, str):
                        try:
                            parsed = json.loads(args)
                            if isinstance(parsed, dict):
                                fn_copy["arguments"] = parsed
                        except json.JSONDecodeError:
                            pass
                    call_copy["function"] = fn_copy
                new_calls.append(call_copy)
            out["tool_calls"] = new_calls
        norm.append(out)
    return norm


def _tokenized_prefix_length(
    tokenizer,
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict]],
    add_generation_prompt: bool,
    template_kwargs: Optional[Dict[str, Any]] = None,
) -> int:
    out = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=False,
        add_generation_prompt=add_generation_prompt,
        padding=False,
        truncation=False,
        **(template_kwargs or {}),
    )
    return len(out.get("input_ids", []))


def _multiturn_assistant_mask(
    tokenizer,
    messages: List[Dict[str, Any]],
    input_ids: List[int],
    tools: Optional[List[Dict]],
    template_kwargs: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """Build an assistant-only loss mask by re-tokenizing each prefix.

    O(n_turns) re-tokenizations. Use when the chat template lacks
    ``{% generation %}`` blocks (i.e. tokenizer can't return assistant_masks).

    ``template_kwargs`` MUST match the kwargs used for the full-sequence
    tokenize call, otherwise prefix lengths won't align with the full
    ``input_ids`` and the mask will be off (e.g. Nemotron's
    ``truncate_history_thinking`` flag changes prefix lengths).
    """
    mask = [0] * len(input_ids)
    found = False
    for idx, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        found = True
        start = _tokenized_prefix_length(
            tokenizer, messages[:idx], tools, add_generation_prompt=True, template_kwargs=template_kwargs
        )
        end = _tokenized_prefix_length(
            tokenizer, messages[: idx + 1], tools, add_generation_prompt=False, template_kwargs=template_kwargs
        )
        for pos in range(min(start, len(mask)), min(end, len(mask))):
            mask[pos] = 1
    if not found:
        raise ValueError("No assistant message in sample")
    return mask


class OversizeError(ValueError):
    """Raised when a tokenized sample exceeds the caller's max_length budget.

    Signals the caller to drop this sample without paying the cost of the
    assistant-mask reconstruction (which is the dominant cost for long
    multi-turn samples).
    """


def tokenize_sample(
    tokenizer,
    row: Dict[str, Any],
    template_kwargs: Optional[Dict[str, Any]] = None,
    max_length: Optional[int] = None,
    chars_per_token: float = 0.0,
) -> Dict[str, List[int]]:
    """Tokenize a single SFT row.

    Parameters
    ----------
    template_kwargs:
        Extra kwargs forwarded to every ``apply_chat_template`` call (both the
        full-sequence tokenize and each prefix re-tokenize used to recover the
        assistant mask). Use this for tokenizer-specific template flags, e.g.
        ``{"truncate_history_thinking": False}`` for the Nemotron chat template
        when you want to keep thinking blocks in earlier assistant turns.
        Unknown kwargs are silently ignored by Jinja templates that don't
        reference them.
    max_length:
        Optional hard cap. After the full-sequence tokenize we know the final
        length; if it exceeds ``max_length`` we raise ``OversizeError`` BEFORE
        running the expensive O(n_turns) assistant-mask reconstruction. The
        caller is expected to catch ``OversizeError`` and drop the sample.
    chars_per_token:
        Cheap pre-filter knob. If > 0, before any tokenize call we estimate the
        token count lower bound as ``total_content_chars / chars_per_token`` and
        reject samples whose estimate already exceeds ``max_length``. Set this
        to the *upper bound* of chars-per-token you've observed in the dataset
        — Nemotron-Cascade-2 SFT data is ~3.7-4.5 chars/token, so 5.0 leaves
        a safety margin. Setting 0 disables the pre-filter. The pre-filter
        only matters when ``max_length`` is also set.

    Returns
    -------
    dict
        ``{"input_ids": [...], "labels": [...]}`` after the standard
        next-token-prediction shift (drop last input_id, drop first label).
        Non-assistant positions in labels are -100.
    """
    if not has_chat_template(tokenizer):
        raise ValueError("Tokenizer lacks chat_template / apply_chat_template")

    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Row missing non-empty `messages` list")

    tools = row.get("tools")
    if tools is not None and not isinstance(tools, list):
        tools = None

    messages = normalize_messages(messages)

    # Char-based pre-filter: cheapest possible oversize reject. Avoids the
    # full apply_chat_template on samples that are obviously too long. The
    # estimate is total_chars / chars_per_token; chars_per_token should be
    # an upper bound for the dataset so this is a token-count *lower* bound.
    if chars_per_token > 0 and max_length is not None:
        total_chars = sum(len(m.get("content", "") or "") for m in messages)
        if total_chars > max_length * chars_per_token:
            raise OversizeError(
                f"chars {total_chars} > {max_length} * {chars_per_token:.1f} (char pre-filter)"
            )

    template_has_generation = GENERATION_REGEX.search(tokenizer.chat_template or "") is not None

    out = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        return_dict=True,
        return_assistant_tokens_mask=template_has_generation,
        add_generation_prompt=False,
        padding=False,
        truncation=False,
        **(template_kwargs or {}),
    )
    input_ids: List[int] = list(out["input_ids"])

    # Oversize fast path: shifted length is len(input_ids) - 1 (possibly +1
    # if EOS gets appended). Use len(input_ids) as a tight lower bound — if
    # this already exceeds max_length, the shifted output certainly will, so
    # bail out before paying the O(n_turns) mask reconstruction cost.
    if max_length is not None and len(input_ids) - 1 > max_length:
        raise OversizeError(f"length {len(input_ids) - 1} > max_length {max_length}")

    if template_has_generation:
        mask = list(out["assistant_masks"])
    else:
        mask = _multiturn_assistant_mask(tokenizer, messages, input_ids, tools, template_kwargs=template_kwargs)

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None and input_ids and input_ids[-1] != eos_id:
        input_ids.append(eos_id)
        mask.append(1)

    if len(input_ids) < 2:
        raise ValueError("Tokenized sample too short")
    if sum(mask) == 0:
        raise ValueError("No assistant tokens after tokenization")

    labels_full = [tok if bool(m) else -100 for tok, m in zip(input_ids, mask)]
    shifted_input_ids = input_ids[:-1]
    shifted_labels = labels_full[1:]
    assert len(shifted_input_ids) == len(shifted_labels)

    return {"input_ids": shifted_input_ids, "labels": shifted_labels}
