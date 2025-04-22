from typing import Optional
from cog import BasePredictor, Input, Path
import torch, soundfile as sf, uuid
from dia.model import Dia

class Predictor(BasePredictor):
    def setup(self):
        self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")  # ≈10 GB VRAM

    def predict(
        self,
        script: str = Input(description="Transcript with [S1] / [S2] tags"),
        seed:   int = Input(default=0, description="Torch seed for deterministic voices")
    ) -> Path:
        torch.manual_seed(seed)
        audio = self.model.generate(script)
        out = f"/tmp/{uuid.uuid4().hex}.wav"
        sf.write(out, audio, 44100)
        return Path(out)
