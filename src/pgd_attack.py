import webbrowser
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from vgg16 import neural_network_model

NUM_CLASSES = 10
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

NORM_MEAN = (0.491, 0.482, 0.446)
NORM_STD = (0.247, 0.243, 0.261)


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def invert(self, x):
        return x * self.std.to(x.device) + self.mean.to(x.device)


def save_results(clean_preds, adv_preds, labels, epsilon, alpha, num_iter):
    lines = []
    lines.append("=" * 45)
    lines.append("PGD ATTACK RESULTS")
    lines.append("=" * 45)
    lines.append(f"Model:       VGG-16 (CIFAR-10)")
    lines.append(f"Attack:      PGD")
    lines.append(f"Epsilon:     {epsilon}")
    lines.append(f"Alpha:       {alpha}")
    lines.append(f"Iterations:  {num_iter}")
    lines.append(f"Norm:        L∞")
    lines.append("")

    clean_acc = (clean_preds == labels).float().mean().item() * 100
    adv_acc = (adv_preds == labels).float().mean().item() * 100
    correct_clean = clean_preds == labels
    fooled = correct_clean & (adv_preds != labels)
    asr = fooled.float().sum() / correct_clean.float().sum().clamp(min=1) * 100

    lines.append(f"Clean Accuracy:    {clean_acc:.2f}%")
    lines.append(f"Adversarial Acc:   {adv_acc:.2f}%")
    lines.append(f"Overall ASR:       {asr:.2f}%")
    lines.append("")
    lines.append(f"{'Class':<12} {'Clean':>8} {'Adv':>8} {'ASR':>8}")
    lines.append("-" * 40)

    for i, cls in enumerate(CIFAR10_CLASSES):
        idx = labels == i
        if idx.sum() == 0:
            continue
        c_acc = (clean_preds[idx] == labels[idx]).float().mean().item() * 100
        a_acc = (adv_preds[idx] == labels[idx]).float().mean().item() * 100
        cm = clean_preds[idx] == labels[idx]
        f = cm & (adv_preds[idx] != labels[idx])
        a = f.float().sum() / cm.float().sum().clamp(min=1) * 100
        lines.append(f"{cls:<12} {c_acc:>7.1f}% {a_acc:>7.1f}% {a:>7.1f}%")

    Path("pgd_results/results.txt").write_text("\n".join(lines))
    print("Saved results.txt")


def load_test_data(batch_size=128):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )
    dataset = datasets.CIFAR10(
        "../data", train=False, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_images, all_labels = [], []
    for images, labels in loader:
        all_images.append(images)
        all_labels.append(labels)
    return torch.cat(all_images), torch.cat(all_labels)


def pgd_attack(
    atk_model, images_01, labels, epsilon=0.03, alpha=0.007, num_iter=10, device="cpu"
):
    images_01 = images_01.to(device)
    labels = labels.to(device)
    loss_fn = nn.CrossEntropyLoss()

    adv = images_01 + torch.empty_like(images_01).uniform_(-epsilon, epsilon)
    adv = adv.clamp(0, 1).detach()

    for _ in range(num_iter):
        adv.requires_grad_(True)
        loss = loss_fn(atk_model(adv), labels)
        grad = torch.autograd.grad(loss, adv)[0]
        adv = adv.detach() + alpha * grad.sign()
        adv = torch.max(torch.min(adv, images_01 + epsilon), images_01 - epsilon)
        adv = adv.clamp(0, 1).detach()

    return adv


def pgd_attack_batched(
    atk_model,
    norm_layer,
    images_norm,
    labels,
    epsilon=0.03,
    alpha=0.007,
    num_iter=10,
    device="cpu",
    batch_size=128,
):
    all_adv = []
    n_batches = (len(images_norm) + batch_size - 1) // batch_size
    for i in range(0, len(images_norm), batch_size):
        img_batch = images_norm[i : i + batch_size]
        lbl_batch = labels[i : i + batch_size]
        img_01 = norm_layer.invert(img_batch).clamp(0, 1)
        adv_01 = pgd_attack(
            atk_model, img_01, lbl_batch, epsilon, alpha, num_iter, device
        )
        all_adv.append(adv_01.cpu())
        print(f"  Batch {i//batch_size + 1}/{n_batches} done")
    return torch.cat(all_adv)


def evaluate(atk_model, images_01, labels, tag="", device="cpu", batch_size=128):
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(images_01), batch_size):
            batch = images_01[i : i + batch_size].to(device)
            preds = atk_model(batch).argmax(dim=1).cpu()
            all_preds.append(preds)
    all_preds = torch.cat(all_preds)
    acc = (all_preds == labels).float().mean().item()
    print(f"[{tag}] Accuracy: {acc*100:.2f}%")
    return all_preds, acc


def evaluate_per_class(clean_preds, adv_preds, labels):
    print(f"\n{'Class':<12} {'Clean':>8} {'Adv':>8} {'ASR':>8}")
    print("-" * 40)
    for i, cls in enumerate(CIFAR10_CLASSES):
        idx = labels == i
        if idx.sum() == 0:
            continue
        clean_acc = (clean_preds[idx] == labels[idx]).float().mean().item()
        adv_acc = (adv_preds[idx] == labels[idx]).float().mean().item()
        correct_mask = clean_preds[idx] == labels[idx]
        fooled = correct_mask & (adv_preds[idx] != labels[idx])
        asr = fooled.float().sum() / correct_mask.float().sum().clamp(min=1)
        print(f"{cls:<12} {clean_acc*100:>7.1f}% {adv_acc*100:>7.1f}% {asr*100:>7.1f}%")


def compute_asr(clean_preds, adv_preds, labels):
    correct_clean = clean_preds == labels
    fooled = correct_clean & (adv_preds != labels)
    asr = fooled.float().sum() / correct_clean.float().sum().clamp(min=1)
    print(f"\nOverall ASR: {asr*100:.2f}%")
    return asr


def compute_perturbation_stats(clean_01, adv_01):
    delta = (adv_01 - clean_01).abs()
    print(f"Avg L2 perturbation:    {delta.norm(p=2, dim=(1,2,3)).mean().item():.4f}")
    print(f"Max L-inf perturbation: {delta.max().item():.4f}")


def visualize_grid(clean_01, adv_01, labels, n_per_class=10):
    def to_numpy(t):
        return t.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()

    n_classes = len(CIFAR10_CLASSES)

    height_ratios = []
    for i in range(n_classes):
        height_ratios += [1, 1]
        if i < n_classes - 1:
            height_ratios.append(0.4)

    fig = plt.figure(figsize=(n_per_class * 0.65, n_classes * 2.0), dpi=320)
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(
        len(height_ratios),
        n_per_class,
        figure=fig,
        hspace=0.05,
        wspace=0.04,
        height_ratios=height_ratios,
        top=0.93,
        bottom=0.01,
        left=0.01,
        right=0.88,
    )

    axes = {}
    for cls_idx in range(n_classes):
        for offset in [0, 1]:
            r = cls_idx * 3 + offset
            for j in range(n_per_class):
                axes[(r, j)] = fig.add_subplot(gs[r, j])

    for cls_idx, cls in enumerate(CIFAR10_CLASSES):
        row_clean = cls_idx * 3
        row_adv = cls_idx * 3 + 1
        idx = (labels == cls_idx).nonzero(as_tuple=True)[0][:n_per_class]

        for j, img_idx in enumerate(idx):
            axes[(row_clean, j)].imshow(
                to_numpy(clean_01[img_idx : img_idx + 1])[0], interpolation="nearest"
            )
            axes[(row_clean, j)].set_facecolor("#00cc66")
            axes[(row_clean, j)].axis("off")

            axes[(row_adv, j)].imshow(
                to_numpy(adv_01[img_idx : img_idx + 1])[0], interpolation="nearest"
            )
            axes[(row_adv, j)].set_facecolor("#ff4444")
            axes[(row_adv, j)].axis("off")

        pos0 = axes[(row_clean, 0)].get_position()
        pos1 = axes[(row_clean, n_per_class - 1)].get_position()
        mid_x = (pos0.x0 + pos1.x1) / 2
        fig.text(
            mid_x,
            pos0.y1 + 0.006,
            cls.upper(),
            fontsize=7,
            fontweight="bold",
            ha="center",
            va="bottom",
            color="#222",
        )

        axes[(row_clean, n_per_class - 1)].text(
            1.04,
            0.5,
            "● clean",
            fontsize=6,
            va="center",
            ha="left",
            transform=axes[(row_clean, n_per_class - 1)].transAxes,
            color="#00aa44",
        )
        axes[(row_adv, n_per_class - 1)].text(
            1.04,
            0.5,
            "● adversarial",
            fontsize=6,
            va="center",
            ha="left",
            transform=axes[(row_adv, n_per_class - 1)].transAxes,
            color="#cc2222",
        )
    results_text = Path("pgd_results/results.txt").read_text()
    clean_acc_line = [l for l in results_text.splitlines() if "Clean Accuracy" in l][0]
    adv_acc_line = [l for l in results_text.splitlines() if "Adversarial Acc" in l][0]
    asr_line = [l for l in results_text.splitlines() if "Overall ASR" in l][0]
    fig.text(
        0.5,
        0.955,
        f"{clean_acc_line.strip()}   |   {adv_acc_line.strip()}   |   {asr_line.strip()}",
        ha="center",
        fontsize=6.5,
        color="#444",
        transform=fig.transFigure,
    )

    fig.text(
        0.5,
        0.99,
        "Clean vs Adversarial by Class",
        ha="center",
        fontsize=11,
        fontweight="bold",
        transform=fig.transFigure,
    )
    fig.text(
        0.5,
        0.972,
        "PGD  |  ε=0.03  |  α=0.007  |  iters=5  |  L∞  |  VGG-16 on CIFAR-10",
        ha="center",
        fontsize=7,
        color="#777",
        transform=fig.transFigure,
    )

    out = Path("pgd_results/clean_vs_adversarial_grid.png")
    plt.savefig(out, dpi=320, bbox_inches="tight")
    print("Saved clean_vs_adversarial_grid.png")
    webbrowser.open(f"file://{out.resolve()}")


def visualize(clean_01, adv_01, labels, clean_preds, adv_preds, n=10):
    def to_numpy(t):
        return t.cpu().permute(0, 2, 3, 1).clamp(0, 1).numpy()

    clean_disp = to_numpy(clean_01[:n])
    adv_disp = to_numpy(adv_01[:n])

    fig, axes = plt.subplots(2, n, figsize=(n * 0.4, 1.2), dpi=320)
    for i in range(n):
        axes[0, i].imshow(clean_disp[i], interpolation="nearest")
        axes[0, i].axis("off")

        axes[1, i].imshow(adv_disp[i], interpolation="nearest")
        axes[1, i].axis("off")

    plt.suptitle("Clean (top) vs Adversarial (bottom)", fontsize=4)
    plt.tight_layout(pad=0.1)
    plt.savefig(
        "pgd_results/clean_vs_adversarial_sample.png", dpi=320, bbox_inches="tight"
    )
    print("Saved clean_vs_adversarial_sample.png")


if __name__ == "__main__":
    WEIGHTS_PATH = "models/cnn.pt"
    ADV_SAVE_PATH = "pgd_results/adv_images.pt"
    EPSILON = 0.03
    ALPHA = 0.007
    NUM_ITER = 5
    BATCH_SIZE = 128
    N_VIS = 10

    Path("pgd_results").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading CIFAR-10...")
    images_norm, labels = load_test_data(batch_size=BATCH_SIZE)

    norm_layer = Normalize(NORM_MEAN, NORM_STD).to(device)

    print("Loading model...")
    base_model = neural_network_model()
    base_model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    base_model = base_model.to(device)
    atk_model = nn.Sequential(norm_layer, base_model).eval()

    images_01 = norm_layer.invert(images_norm.to(device)).clamp(0, 1).cpu()

    clean_preds, _ = evaluate(atk_model, images_01, labels, tag="Clean", device=device)

    if Path(ADV_SAVE_PATH).exists():
        print("Loading saved adversarial images...")
        adv_01 = torch.load(ADV_SAVE_PATH)
    else:
        print(f"\nRunning PGD attack (ε={EPSILON}, α={ALPHA}, iters={NUM_ITER})...")
        adv_01 = pgd_attack_batched(
            atk_model,
            norm_layer,
            images_norm,
            labels,
            EPSILON,
            ALPHA,
            NUM_ITER,
            device,
            BATCH_SIZE,
        )
        torch.save(adv_01, ADV_SAVE_PATH)
        print(f"Saved adversarial images to {ADV_SAVE_PATH}")

    adv_preds, _ = evaluate(atk_model, adv_01, labels, tag="Adversarial", device=device)

    evaluate_per_class(clean_preds, adv_preds, labels)
    compute_asr(clean_preds, adv_preds, labels)
    save_results(clean_preds, adv_preds, labels, EPSILON, ALPHA, NUM_ITER)
    compute_perturbation_stats(images_01, adv_01)

    visualize(images_01, adv_01, labels, clean_preds, adv_preds, n=N_VIS)
    visualize_grid(images_01, adv_01, labels, n_per_class=10)
