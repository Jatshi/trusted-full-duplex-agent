"""Narrow, session-local resume exception, not KV erasure or an action policy."""
import math
import re
import time


_RESUME = re.compile(
    r'(?:请)?(?:你)?(?:继续|接着)(?:讲|说)?(?:你)?(?:刚才|之前)(?:的|说的|讲的)?'
    r'(?:那个|这个)?(?:故事|话题|内容|回答)(?:吧|一下)?[。！!?？]*')


class InterruptedResponseContext:
    """Retain only a response ID after observed cancellation, for at most 120s."""

    def __init__(self, *, clock=time.monotonic):
        self.clock = clock
        self.active = None
        self.safe = False
        self.has_text = False
        self.pending = None
        self.cancelled_at = None
        self.resolved_count = 0

    def begin(self, response_id, *, safe):
        self.active = response_id
        self.safe = bool(safe)
        self.has_text = False
        self.pending = self.cancelled_at = None

    def output_text(self, text):
        self.has_text = self.has_text or bool(str(text).strip())

    def cancel(self, response_id):
        if response_id is not None and response_id == self.active:
            if self.safe and self.has_text:
                self.pending = response_id
                self.cancelled_at = self.clock()
            self.active = None

    def finish(self):
        self.active = self.pending = self.cancelled_at = None
        self.safe = self.has_text = False

    def resolve(self, text, *, action, risk, confidence, min_confidence=.6):
        age = self.clock() - self.cancelled_at if self.cancelled_at is not None else float('inf')
        eligible = (self.pending is not None and 0 <= age <= 120 and action == 'clarify'
                    and math.isfinite(risk) and risk == 0 and math.isfinite(confidence)
                    and math.isfinite(min_confidence) and .45 <= min_confidence <= 1
                    and min_confidence <= confidence <= 1 and len(text) <= 40
                    and _RESUME.fullmatch(text.strip()) is not None)
        if eligible:
            self.pending = self.cancelled_at = None
            self.resolved_count += 1
        return bool(eligible)

    def metrics(self):
        age = self.clock() - self.cancelled_at if self.cancelled_at is not None else float('inf')
        return {'tfd_resume_context_enabled': True,
                'tfd_resume_context_available': self.pending is not None and 0 <= age <= 120,
                'tfd_resume_resolved_count': self.resolved_count,
                'tfd_resume_clears_model_kv': False}
