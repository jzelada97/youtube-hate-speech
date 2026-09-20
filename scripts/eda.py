"""EDA reproducible del dataset crudo. Ejecutar: python scripts/eda.py

Se implementa como script (no notebook) para que sea reproducible con un solo comando (NFR-3)
y quede como parte del pipeline de análisis, no como exploración descartable.
Genera figuras en reports/figures/ y estadísticas en stdout.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from hatedet.data.loader import load_raw_comments
from hatedet.data.schema import LABEL_COLUMNS, TARGET_COLUMN, TEXT_COLUMN, VIDEO_ID_COLUMN

FIGURES_DIR = Path("reports/figures")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df = load_raw_comments()

    print(f"Filas tras dedup: {len(df)}")
    print(f"Videos unicos: {df[VIDEO_ID_COLUMN].nunique()}")
    print()
    print("Balance de clases (% positivos):")
    print((df[LABEL_COLUMNS].mean() * 100).round(2).sort_values(ascending=False))
    print()
    print("Longitud de texto (caracteres):")
    lengths = df[TEXT_COLUMN].str.len()
    print(lengths.describe())

    # Figura 1: balance de clases
    fig, ax = plt.subplots(figsize=(8, 5))
    (df[LABEL_COLUMNS].mean() * 100).sort_values().plot.barh(ax=ax, color="steelblue")
    ax.set_xlabel("% positivos")
    ax.set_title("Balance de clases (todas las etiquetas)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "class_balance.png", dpi=120)
    plt.close(fig)

    # Figura 2: distribucion de longitud de texto por target
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(
        data=df.assign(length=lengths),
        x="length",
        hue=TARGET_COLUMN,
        bins=40,
        log_scale=(False, True),
        ax=ax,
    )
    ax.set_xlim(0, 1000)
    ax.set_title(f"Longitud de comentario por {TARGET_COLUMN}")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "text_length_by_target.png", dpi=120)
    plt.close(fig)

    # Figura 3: comentarios por video, coloreado por tasa de odio
    fig, ax = plt.subplots(figsize=(8, 5))
    by_video = df.groupby(VIDEO_ID_COLUMN).agg(
        n=(TEXT_COLUMN, "size"), hate_rate=(TARGET_COLUMN, "mean")
    ).sort_values("n", ascending=False)
    by_video["n"].plot.bar(ax=ax, color=plt.cm.Reds(by_video["hate_rate"] / by_video["hate_rate"].max()))
    ax.set_ylabel("N comentarios")
    ax.set_title("Comentarios por video (color = tasa de odio)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "comments_per_video.png", dpi=120)
    plt.close(fig)

    print()
    print("Distribucion de comentarios por video:")
    print(by_video)
    print()
    print(f"Figuras guardadas en {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
