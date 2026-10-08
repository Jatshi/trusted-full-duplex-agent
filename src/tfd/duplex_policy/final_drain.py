"""Finite replay lifecycle: EOF, a new final response, then natural turn end.

This tracks measurement only; it never forces an end token or changes logits.
"""
class FinalDrain:
    def __init__(self):
        self.started=False
        self.complete=False

    def observe(self, *, eof, is_listen, active_before, end_of_turn):
        if not eof:
            return False
        if not is_listen and not active_before:
            self.started=True
        if self.started and end_of_turn:
            self.complete=True
        return self.complete
