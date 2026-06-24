import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataset import GrainDataset
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = GrainDataset(
    manifest_path="manifests/train.csv",
    class_mapping_path="configs/classes.json",
    transform=transform,
)

print("Dataset size:", len(dataset))
print("Class mapping:", dataset.class_to_idx)

image, label = dataset[0]

print("First image tensor shape:", tuple(image.shape))
print("First image label ID:", label)
print("First image label name:", dataset.df.iloc[0]["label"])
print("Dataset loader test passed.")
