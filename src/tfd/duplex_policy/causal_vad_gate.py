"""Causal fixed Silero speech confirmation; no oracle phase or future samples."""
import numpy as np
import torch


class CausalVadGate:
    def __init__(self, model):
        self.model=model
        self.reset()

    def reset(self):
        self.model.reset_states()
        self.buffer=np.empty(0,dtype=np.float32)
        self.consecutive=0

    def feed(self, audio):
        audio=np.asarray(audio,dtype=np.float32)
        if audio.ndim!=1 or not np.isfinite(audio).all():
            raise ValueError('finite mono input required')
        self.buffer=np.concatenate((self.buffer,audio))
        detected=False
        while len(self.buffer)>=512:
            frame=torch.from_numpy(self.buffer[:512].copy())
            self.buffer=self.buffer[512:]
            with torch.no_grad():
                probability=float(self.model(frame,16000))
            if not 0<=probability<=1:
                raise ValueError('invalid VAD probability')
            self.consecutive=self.consecutive+1 if probability>=.5 else 0
            detected=detected or self.consecutive>=3
        return detected
