import argparse
import json

import torch
from PIL import Image
from torchvision import transforms

from model import DualBranchCoAtNetPVTv2Classifier


CLASS_NAMES = ["fake", "real"]  # change if your label order differs


def build_transform(img_size: int):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def load_model(checkpoint_path: str, device: torch.device, dropout: float, elu_alpha: float):
    model = DualBranchCoAtNetPVTv2Classifier(
        dropout=dropout,
        elu_alpha=elu_alpha,
    )
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def predict_image(model, image_path: str, device: torch.device, img_size: int):
    image = Image.open(image_path).convert("RGB")
    tensor = build_transform(img_size)(image).unsqueeze(0).to(device)

    logits = model(tensor)
    probs = torch.softmax(logits, dim=1)[0].cpu()

    pred_idx = int(torch.argmax(probs).item())
    confidence = float(probs[pred_idx].item())

    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": confidence,
        "probabilities": {
            CLASS_NAMES[i]: float(probs[i].item()) for i in range(len(CLASS_NAMES))
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--elu_alpha", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = load_model(args.checkpoint, device, args.dropout, args.elu_alpha)
    result = predict_image(model, args.image, device, args.img_size)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
