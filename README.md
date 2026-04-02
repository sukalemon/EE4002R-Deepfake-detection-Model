# Hybrid Deepfake Detector

A dual-branch deepfake image classifier built with PyTorch and timm.

## Architecture

Input image is processed by two parallel branches:

- Branch 1: CoAtNet -> PVTv2Stage -> GELU
- Branch 2: CoAtNet -> PVTv2Stage -> ELU

The two branch features are:
- normalized with LayerNorm
- scaled by learnable branch weights
- concatenated
- passed through a classifier head

## Repo structure

```text
hybrid-deepfake-detector/
├── model.py
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── checkpoints/
```

## Installation

```bash
pip install -r requirements.txt
```

## Dataset format

The dataset should follow ImageFolder format:

```text
data/
├── fake/
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
└── real/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

## Training

Example:

```bash
python train.py \
  --data_dir /path/to/data \
  --save_dir checkpoints \
  --save_name best_model.pth \
  --img_size 224 \
  --batch_size 8 \
  --epochs 8 \
  --warmup_epochs 3 \
  --lr 1e-5 \
  --dropout 0.4
```

## Inference

Example:

```bash
python predict.py \
  --image /path/to/test.jpg \
  --checkpoint checkpoints/best_model.pth
```

## Output format

Example:

```json
{
  "predicted_class": "fake",
  "confidence": 0.9132,
  "probabilities": {
    "fake": 0.9132,
    "real": 0.0868
  }
}
```

## Notes

- If your label order is different, update `CLASS_NAMES` in `predict.py`.
- If GPU memory is limited, reduce `--batch_size`.
- The best checkpoint is chosen by validation ROC-AUC.
