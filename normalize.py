"""IPA -> Proto-Germanic Piper voice input.

The model was trained on labels produced by exactly this transform. Skipping
it does not raise an error; it produces confident mispronunciation.
"""

import unicodedata as ud

STRESS = "\u02c8\u02cc"
NON_SYLLABIC = "\u032f"
SYLLABLE_BREAKS = ".\u2027\u00b7|"
NASAL_DECOMP = {"\u0129": "i\u0303", "\u0169": "u\u0303"}
PRE_SUBS = {"\u02b7": "w", "\u1d5d": "\u03b2"}
VOWELS = set("aeiouy\u0251\u0250\u0252\u00e6\u0254\u0259\u025b\u025c\u0258"
             "\u025e\u0268\u026a\u026f\u00f8\u0275\u0153\u0276\u0289\u028a"
             "\u028c\u028f\u0264\u1d7b")


def restress(s):
    """Move each stress mark right, to immediately before its vowel."""
    src, out, i = list(s), [], 0
    while i < len(src):
        ch = src[i]
        if ch in STRESS:
            j, onset = i + 1, []
            while j < len(src) and src[j] not in VOWELS and src[j] not in STRESS:
                if ud.category(src[j]) == "Mn" and not onset:
                    break
                onset.append(src[j])
                j += 1
            if j < len(src) and src[j] in VOWELS:
                out.extend(onset)
                out.append(ch)
                i = j
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def add_primary_stress(s):
    """Unstressed entries (suffixes) get stress on their first vowel.

    The voice saw almost only stressed words, so an unstressed vowel-initial
    form comes out clipped to near-inaudibility.
    """
    if any(c in STRESS for c in s):
        return s
    for i, ch in enumerate(s):
        if ch in VOWELS:
            return s[:i] + "\u02c8" + s[i:]
    return s


def normalize(ipa, force_stress=True):
    s = ud.normalize("NFC", (ipa or "").strip())
    for k, v in NASAL_DECOMP.items():
        s = s.replace(k, v)
    for k, v in PRE_SUBS.items():
        s = s.replace(k, v)
    s = "".join(c for c in s if c not in SYLLABLE_BREAKS and c != NON_SYLLABIC)
    if force_stress:
        s = add_primary_stress(s)
    return restress(s)


def to_ids(phonemes, id_map):
    """Piper's convention: BOS, PAD, then each phoneme followed by PAD, EOS.

    The PAD after BOS is not optional -- without it every token is shifted one
    position and the duration predictor inserts or drops syllables.
    """
    ids = list(id_map["^"]) + list(id_map["_"])
    unknown = []
    for p in phonemes:
        if p in id_map:
            ids.extend(id_map[p])
            ids.extend(id_map["_"])
        else:
            unknown.append(p)
    ids.extend(id_map["$"])
    return ids, unknown
