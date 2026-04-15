# Adversarial Attacks on VGG-16: FGSM & PGD

## Setup (Conda)

1. Install Miniconda or Anaconda from https://docs.conda.io/en/latest/miniconda.html
2. Create the environment:

```bash
   conda create -n finalProj python=3.10 -y
   conda activate finalProj
   pip install -r requirements.txt
```

## Training the Model

Train the custom VGG-16 on CIFAR-10 (saves to `src/models/cnn.pt`):

```bash
python src/vgg16.py --train-model True
```

## Running Attacks

All scripts should be run from the repo root with the conda environment active.

### PGD Attack

```bash
python src/pgd_attack.py
```

Results saved to `src/pgd_results/`

### FGSM Attack

```bash
python src/fgsm_attack.py
```

Results saved to `src/fgsm_results/`

## Output

Each attack produces:

- `results.txt` — clean accuracy, adversarial accuracy, and ASR per class
- `clean_vs_adversarial_grid.png` — visual comparison of clean vs perturbed images
- `clean_vs_adversarial_sample.png` — sample of 10 clean/adversarial pairs

## Hyperparameters

| Parameter     | PGD   | FGSM |
| ------------- | ----- | ---- |
| ε (epsilon)   | 0.03  | 0.03 |
| α (step size) | 0.007 | -    |
| Iterations    | 5     | 1    |
| Norm          | L∞    | L∞   |
