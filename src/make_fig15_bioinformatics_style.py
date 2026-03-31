from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUTDIR = Path("outputs/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "input": "#DCEBFA",
    "prep": "#E8E2FB",
    "split": "#EDEDED",
    "expert": "#F8E39B",
    "datadriven": "#F6C6E3",
    "ml": "#CFEFD6",
    "eval": "#FFD7CC",
    "unknown": "#FDE7C7",
    "text": "#1F2937",
    "muted": "#4B5563",
    "edge": "#1F2937",
    "arrow_main": "#374151",
    "arrow_expert": "#C58B00",
    "arrow_data": "#C0268C",
    "arrow_ml": "#1F9D55",
    "arrow_unknown": "#D97706",
}

FIG_W = 18
FIG_H = 8

def rounded_box(ax, x, y, w, h, title, subtitle="",
                facecolor="#FFFFFF", edgecolor="#1F2937",
                linewidth=2.0, shadow=True,
                title_size=15, subtitle_size=11,
                title_y=0.64, subtitle_y=0.33):
    if shadow:
        ax.add_patch(FancyBboxPatch(
            (x + 0.008, y - 0.010), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.03",
            linewidth=0,
            facecolor="black",
            alpha=0.08,
            zorder=1
        ))

    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.03",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=2
    )
    ax.add_patch(patch)

    ax.text(
        x + w/2, y + h*title_y, title,
        ha="center", va="center",
        fontsize=title_size, fontweight="bold",
        color=COLORS["text"], zorder=3
    )

    if subtitle:
        ax.text(
            x + w/2, y + h*subtitle_y, subtitle,
            ha="center", va="center",
            fontsize=subtitle_size,
            color=COLORS["text"],
            linespacing=1.15,
            zorder=3
        )

def arrow(ax, x1, y1, x2, y2, color, lw=2.8, rad=0.0):
    arr = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="->",
        mutation_scale=20,
        linewidth=lw,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
        zorder=4
    )
    ax.add_patch(arr)

def label_chip(ax, x, y, text, fc, w=0.078, h=0.04):
    chip = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=0,
        facecolor=fc,
        zorder=5
    )
    ax.add_patch(chip)
    ax.text(
        x + w/2, y + h/2, text,
        ha="center", va="center",
        fontsize=9.5, color=COLORS["text"], zorder=6
    )

def make_fig15():
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 1.24)
    ax.set_ylim(0, 1.00)
    ax.axis("off")

    # Title only
    ax.text(
        0.62, 0.945,
        "Overview of the expert-rule fuzzy, data-driven fuzzy, and machine-learning analysis workflow",
        ha="center", va="bottom",
        fontsize=20, fontweight="bold", color=COLORS["text"]
    )

    # Left-to-right boxes
    rounded_box(
        ax, 0.05, 0.42, 0.16, 0.14,
        "Input dataset",
        "clinical + tumor\nfeatures",
        facecolor=COLORS["input"],
        title_size=16, subtitle_size=11.5
    )

    rounded_box(
        ax, 0.28, 0.42, 0.16, 0.14,
        "Preprocessing",
        "encoding + cleaning\nand target definition",
        facecolor=COLORS["prep"],
        title_size=16, subtitle_size=11.5
    )

    rounded_box(
        ax, 0.51, 0.42, 0.16, 0.14,
        "Repeated stratified",
        "80/20 splits",
        facecolor=COLORS["split"],
        title_size=16, subtitle_size=12
    )

    # Branch boxes
    rounded_box(
        ax, 0.80, 0.69, 0.18, 0.11,
        "Expert-rule fuzzy",
        "interpretable rule base",
        facecolor=COLORS["expert"],
        title_size=16, subtitle_size=11.5
    )

    rounded_box(
        ax, 0.80, 0.47, 0.18, 0.11,
        "Data-driven fuzzy",
        "Wang–Mendel + Mamdani",
        facecolor=COLORS["datadriven"],
        title_size=16, subtitle_size=11.5
    )

    rounded_box(
        ax, 0.80, 0.25, 0.18, 0.11,
        "Machine learning",
        "LR / SVM / RF / GB",
        facecolor=COLORS["ml"],
        title_size=16, subtitle_size=11.5
    )

    rounded_box(
        ax, 1.07, 0.42, 0.14, 0.14,
        "Evaluation",
        "Accuracy / F1 / BAcc\nROC / PR / confusion",
        facecolor=COLORS["eval"],
        title_size=16, subtitle_size=11,
        subtitle_y=0.30
    )

    rounded_box(
        ax, 0.56, 0.08, 0.32, 0.10,
        "Unknown-Stage secondary analyses",
        "initial prediction / self-training / label propagation",
        facecolor=COLORS["unknown"],
        title_size=14, subtitle_size=10.8,
        title_y=0.63, subtitle_y=0.29
    )

    # Main arrows
    arrow(ax, 0.21, 0.49, 0.28, 0.49, COLORS["arrow_main"])
    arrow(ax, 0.44, 0.49, 0.51, 0.49, COLORS["arrow_main"])

    # Split -> methods
    arrow(ax, 0.67, 0.49, 0.80, 0.745, COLORS["arrow_expert"])
    arrow(ax, 0.67, 0.49, 0.80, 0.525, COLORS["arrow_data"])
    arrow(ax, 0.67, 0.49, 0.80, 0.305, COLORS["arrow_ml"])
    arrow(ax, 0.60, 0.41, 0.73, 0.18, COLORS["arrow_unknown"])

    # Methods -> evaluation
    arrow(ax, 0.98, 0.745, 1.07, 0.505, COLORS["arrow_expert"])
    arrow(ax, 0.98, 0.525, 1.07, 0.495, COLORS["arrow_data"])
    arrow(ax, 0.98, 0.305, 1.07, 0.485, COLORS["arrow_ml"])

    # Chips
    label_chip(ax, 0.095, 0.57, "Data", "#BFD8F4")
    label_chip(ax, 0.325, 0.57, "QC", "#D7CCF7")
    label_chip(ax, 0.555, 0.57, "CV", "#DDDDDD")
    label_chip(ax, 0.865, 0.82, "Rules", "#F6D56A")
    label_chip(ax, 0.865, 0.60, "WM", "#F2A9D2")
    label_chip(ax, 0.865, 0.38, "ML", "#A9E2B4")
    label_chip(ax, 1.10, 0.57, "Perf", "#FFC1B2")

    fig.tight_layout(pad=1.4)
    fig.savefig(OUTDIR / "Fig15_workflow_schematic.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(OUTDIR / "Fig15_workflow_schematic.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)

if __name__ == "__main__":
    make_fig15()
    print("Saved stylized Fig15 to outputs/figures/")
