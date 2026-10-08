"""Mechanical copy clues, explicitly not a semantic correctness grader."""
import re
from difflib import SequenceMatcher


def copy_diagnostic(output, source):
    tokens = lambda text: re.findall(r"[a-z0-9]+", text.lower())
    generated, caller = tokens(output), tokens(source)
    span = SequenceMatcher(None, generated, caller, autojunk=False).find_longest_match().size
    source_grams = {tuple(caller[i:i+4]) for i in range(max(0, len(caller)-3))}
    out_grams = [tuple(generated[i:i+4]) for i in range(max(0, len(generated)-3))]
    fraction = sum(gram in source_grams for gram in out_grams) / len(out_grams) if out_grams else 0.
    return dict(output_words=len(generated), longest_copied_span_words=span,
                copied_fourgram_fraction=fraction, copy_review_flag=span >= 8 and fraction >= .5,
                semantic_pass=None,
                limitation="Mechanical review flag only; quoting may be valid, low overlap may still be wrong")
