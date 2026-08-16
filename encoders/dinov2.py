import torch

from encoders.base import BaseEncoder


class DinoV2Encoder(BaseEncoder):
    def __init__(
        self,
        model_name="dinov2_vits14",
        device=None,
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.device = torch.device(device)

        print(f"Loading {model_name} on {self.device}...")

        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
        )

        self.model.eval()
        self.model.to(self.device)

    @torch.no_grad()
    def encode(self, images):
        images = images.to(self.device)

        embeddings = self.model(images)

        return embeddings.cpu()