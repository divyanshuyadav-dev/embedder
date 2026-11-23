import cv2
import torch
import numpy as np

def get_embeddings(image_path:str, model_path:str):
    # Load image
    img = cv2.imread(image_path)

    if img is None:
        raise FileNotFoundError("Image not found")

    # OpenCV loads BGR; convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Resize
    img = cv2.resize(img, (112, 112))

    # Normalize pixel values to [-1, 1]
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0

    # Convert HWC -> CHW
    img = np.transpose(img, (2, 0, 1))

    # Add batch dimension
    img = torch.from_numpy(img).unsqueeze(0)

    print("Input shape:", img.shape)

    model = torch.jit.load(model_path)
    model.eval()

    with torch.no_grad():
        embedding = model(img)

    print("Embedding shape:", embedding.shape)
    print(embedding)
    return torch.tensor(embedding)

model_path = "mobilefacenet_scripted.pt"

x = get_embeddings("inference_media/cattle-400/IMG20230924150617_01.jpg", model_path=model_path)
# print(x)
y = get_embeddings("inference_media/cattle-400/IMG20230924150607_01.jpg", model_path=model_path)
# print(y)

cos_layer = torch.nn.CosineSimilarity()
print(cos_layer(x, y))
