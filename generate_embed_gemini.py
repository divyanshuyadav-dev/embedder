import cv2
import torch
import torch.nn.functional as F
import numpy as np

# Import MobileFaceNet from your training/model file
from train import MobileFaceNet


class CattleEmbedder:
    def __init__(self, checkpoint_path: str, device: str = None):
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        ckpt = torch.load(checkpoint_path, map_location=self.device)
        embedding_dim = ckpt.get("embedding_dim", 512)

        self.model = MobileFaceNet(embedding_size=embedding_dim).to(self.device)
        self.model.load_state_dict(ckpt["backbone"])
        self.model.eval()

    def preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (112, 112), interpolation=cv2.INTER_LINEAR)

        # Matches torchvision Normalize(mean=[0.5]*3, std=[0.5]*3)
        img = (img.astype(np.float32) / 127.5) - 1.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        return torch.from_numpy(img).unsqueeze(0).to(self.device)

    @torch.inference_mode()
    def get_embedding(self, image_or_path) -> torch.Tensor:
        if isinstance(image_or_path, str):
            img = cv2.imread(image_or_path)
            if img is None:
                raise FileNotFoundError(f"Image not found: {image_or_path}")
        else:
            img = image_or_path

        tensor = self.preprocess(img)
        embedding = self.model(tensor)
        return F.normalize(embedding, p=2, dim=1)


if __name__ == "__main__":
    embedder = CattleEmbedder("cattle_checkpoints/best.pth")

    x = embedder.get_embedding("inference_media/cattle-400/IMG20230924150617_01.jpg")
    y = embedder.get_embedding("inference_media/cattle-400/IMG20230924150607_01.jpg")

    # Since both are L2-normalized, dot product == cosine similarity
    similarity = (x @ y.T).item()
    print(f"Cosine similarity: {similarity:.4f}")
