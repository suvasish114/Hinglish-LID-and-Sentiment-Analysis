"""Experiment 2 - switch-point features derived from word-level LID tags.

    Tokens:  Movie  bahut  acchi  thi  but  ending  was  terrible
    LID:       e      h      h     h    e     e       e     e
    Switch:    0      1      0     0    1     0       0     0

switch_i = 1 if LID_i != LID_{i-1} else 0, switch_0 = 0 by convention. Tokens tagged `o` (punctuation,
mentions, emoji, numbers) are transparent: they are never a switch themselves and do not reset the previous
language, so `h o e` is one switch (on `e`). Pass ignore_other=False for the naive definition.
"""

from typing import List, Tuple

OTHER = "o"
LANGS = ("h", "e")


def switch_points(tags: List[str], ignore_other: bool = True) -> List[int]:
    switches, prev = [0] * len(tags), None
    for i, tag in enumerate(tags):
        if ignore_other and tag == OTHER:
            continue
        if prev is not None and tag != prev:
            switches[i] = 1
        prev = tag
    return switches


def switch_directions(tags: List[str], ignore_other: bool = True) -> List[int]:
    """0 = no switch, 1 = Hindi -> English, 2 = English -> Hindi (same convention as switch_points)."""
    out, prev = [0] * len(tags), None
    for i, tag in enumerate(tags):
        if ignore_other and tag == OTHER:
            continue
        if prev is not None and tag != prev:
            out[i] = 1 if (prev == "h" and tag == "e") else 2
        prev = tag
    return out


def switch_distances(switches: List[int], max_bucket: int = 3) -> List[int]:
    """Distance (in words) to the nearest switch point, bucketed to 0..max_bucket (3 = "3+").
    0 = the word is a switch, 1 = adjacent, ...; max_bucket also when the tweet has no switch."""
    n = len(switches)
    dist, last = [n + max_bucket + 1] * n, None
    for i, s in enumerate(switches):                 # left-to-right
        if s:
            last = i
        if last is not None:
            dist[i] = i - last
    last = None
    for i in range(n - 1, -1, -1):                   # right-to-left
        if switches[i]:
            last = i
        if last is not None:
            dist[i] = min(dist[i], last - i)
    return [min(d, max_bucket) for d in dist]


def upcoming_switch_labels(switches: List[int]) -> List[int]:
    """Label of word i = 1 if the NEXT word starts a new language (the prediction task of
    "Beyond Detection: Predicting Code-Switch Points"); the last word gets 0."""
    return switches[1:] + [0]


def language_segments(tags: List[str], ignore_other: bool = True) -> List[Tuple[str, List[int]]]:
    """Maximal monolingual runs (language, word indices) delimited by the switch points; `o` words
    are attached to the run they occur in (leading `o` words go to the first run)."""
    segments, cur_lang, cur = [], None, []
    for i, tag in enumerate(tags):
        lang = None if (ignore_other and tag == OTHER) else tag
        if lang is not None and cur_lang is not None and lang != cur_lang:
            segments.append((cur_lang, cur))
            cur = []
        if lang is not None:
            cur_lang = lang
        cur.append(i)
    if cur:
        segments.append((cur_lang if cur_lang is not None else OTHER, cur))
    return segments


def language_view(tokens: List[str], tags: List[str], lang: str) -> List[str]:
    """Approach 4 - the `lang` view of a tweet: its words that belong to the `lang` segments, in order.
    Falls back to the whole tweet when it has no segment of that language."""
    idx = [i for seg_lang, ids in language_segments(tags) if seg_lang == lang for i in ids]
    return [tokens[i] for i in idx] if idx else list(tokens)


def align_to_subwords(word_feats: List[int], word_index, pad_value: int) -> List[int]:
    """Spread a per-word feature over the sub-words (special tokens -> pad_value)."""
    return [pad_value if w is None else word_feats[w] for w in word_index]


def n_switches(tags: List[str]) -> int:
    return sum(switch_points(tags))
