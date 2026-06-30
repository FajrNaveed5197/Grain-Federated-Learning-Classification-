from pathlib import Path
import pandas as pd
from PIL import Image
import torch
from torchvision import models, transforms

manifest = Path("/scratch/project_2019649/grain_research/manifests/train.csv")
df = pd.read_csv(manifest)

row = df.iloc[0]
image = Image.open(row["path"]).convert("RGB")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

tensor = transform(image).unsqueeze(0)

print("Manifest rows:", len(df))
print("First label:", row["label"])
print("First path:", row["path"])
print("Image tensor shape:", tuple(tensor.shape))
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")

model = models.mobilenet_v2(
    weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
)
model.classifier[1] = torch.nn.Linear(
    model.classifier[1].in_features,
    8,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
output = model(tensor.to(device))

print("Model output shape:", tuple(output.shape))
print("CSC GPU/data test passed.")
