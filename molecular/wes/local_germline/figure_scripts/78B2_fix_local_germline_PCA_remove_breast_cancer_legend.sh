#!/usr/bin/env bash

# Load portable molecular-analysis paths.
_REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || true)"
if [[ -z "${_REPO_ROOT}" ]]; then
    echo "ERROR: Run this script from within the bc-fuzzy-ml-repro repository." >&2
    exit 1
fi
# shellcheck source=/dev/null
source "${_REPO_ROOT}/molecular/config/load_config.sh"

set -euo pipefail

cd $PROJECT_ROOT

PROJECT="$PROJECT_ROOT"
PAPER="$PAPER_RESULTS_ROOT"

TABLE_DIR="$PAPER/05_local_germline_WES/01_tables"
FIG_DIR="$PAPER/publication_figures/local_germline_WES"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$FIG_DIR" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/78B2_fix_local_germline_PCA_remove_breast_cancer_legend_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 78B2: FIX LOCAL GERMLINE PCA LEGEND ==="
date

PCA_TABLE="$TABLE_DIR/local_wes_pca_with_metadata.tsv"

if [[ ! -s "$PCA_TABLE" ]]; then
  echo "ERROR: PCA table not found:"
  echo "$PCA_TABLE"
  exit 1
fi

echo "Input:"
echo "$PCA_TABLE"

# Backup old figures before overwrite
for f in \
  "$FIG_DIR/Fig_LocalWES_01_PCA_PC1_PC2_population_structure.png" \
  "$FIG_DIR/Fig_LocalWES_01_PCA_PC1_PC2_population_structure.pdf" \
  "$FIG_DIR/Fig_LocalWES_01B_PCA_PC1_PC3_population_structure.png" \
  "$FIG_DIR/Fig_LocalWES_01B_PCA_PC1_PC3_population_structure.pdf"
do
  if [[ -s "$f" && ! -s "$f.bak_before_78B2" ]]; then
    cp -av "$f" "$f.bak_before_78B2"
  fi
done

export PCA_TABLE FIG_DIR STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
})

pca_table <- Sys.getenv("PCA_TABLE")
fig_dir <- Sys.getenv("FIG_DIR")
status <- Sys.getenv("STATUS")

df <- read_tsv(pca_table, show_col_types = FALSE)

# Flexible column detection
sample_col <- intersect(c("sample_id", "sample", "IID", "id"), names(df))[1]
group_col  <- intersect(c("group", "Group", "population", "cohort"), names(df))[1]
sex_col    <- intersect(c("sex", "Sex", "sex_from_id", "sex_label"), names(df))[1]

pc1_col <- intersect(c("PC1", "PC_1", "pc1"), names(df))[1]
pc2_col <- intersect(c("PC2", "PC_2", "pc2"), names(df))[1]
pc3_col <- intersect(c("PC3", "PC_3", "pc3"), names(df))[1]

needed <- c(sample_col, group_col, sex_col, pc1_col, pc2_col, pc3_col)
if (any(is.na(needed))) {
  cat("Columns detected:\n")
  print(names(df))
  stop("Required PCA/metadata columns could not be detected.")
}

plot_df <- df %>%
  rename(
    sample_id = all_of(sample_col),
    group = all_of(group_col),
    sex = all_of(sex_col),
    PC1 = all_of(pc1_col),
    PC2 = all_of(pc2_col),
    PC3 = all_of(pc3_col)
  ) %>%
  mutate(
    group = case_when(
      grepl("^H|Hadza", sample_id, ignore.case = TRUE) ~ "Hadza",
      grepl("^C|control|Tanzanian", sample_id, ignore.case = TRUE) ~ "Tanzanian control",
      TRUE ~ as.character(group)
    ),
    sex = case_when(
      grepl("^F|female", sex, ignore.case = TRUE) ~ "Female",
      grepl("^M|male", sex, ignore.case = TRUE) ~ "Male",
      TRUE ~ as.character(sex)
    )
  ) %>%
  filter(group %in% c("Hadza", "Tanzanian control")) %>%
  mutate(
    group = factor(group, levels = c("Hadza", "Tanzanian control")),
    sex = factor(sex, levels = c("Female", "Male"))
  ) %>%
  droplevels()

# PC variance labels: use values from previous figure unless exact values are available elsewhere
xlab12 <- "PC1 (6.9%)"
ylab12 <- "PC2 (6.5%)"
xlab13 <- "PC1 (6.9%)"
ylab13 <- "PC3"

group_cols <- c(
  "Hadza" = "#009E73",
  "Tanzanian control" = "#0072B2"
)

sex_shapes <- c(
  "Female" = 16,
  "Male" = 17
)

theme_pca <- function(base_size = 12) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 5),
      plot.subtitle = element_text(size = base_size + 1, margin = margin(b = 8)),
      axis.title = element_text(face = "bold", size = base_size + 1),
      axis.text = element_text(color = "grey20"),
      axis.line = element_line(linewidth = 0.45, color = "grey20"),
      legend.position = "top",
      legend.box = "horizontal",
      legend.justification = "left",
      legend.title = element_text(face = "bold"),
      legend.text = element_text(size = base_size),
      legend.margin = margin(t = 2, r = 2, b = 8, l = 0),
      plot.margin = margin(t = 14, r = 18, b = 14, l = 14)
    )
}

make_pca <- function(y_col, ylab, out_prefix) {
  p <- ggplot(plot_df, aes(x = PC1, y = .data[[y_col]])) +
    stat_ellipse(
      aes(color = group),
      geom = "path",
      linewidth = 0.75,
      alpha = 0.8,
      type = "norm",
      show.legend = FALSE
    ) +
    geom_point(
      aes(color = group, shape = sex),
      size = 3.5,
      alpha = 0.92,
      stroke = 0.4
    ) +
    scale_color_manual(
      name = "Group",
      values = group_cols,
      drop = TRUE
    ) +
    scale_shape_manual(
      name = "Sex",
      values = sex_shapes,
      drop = TRUE
    ) +
    labs(
      title = "Local germline WES population structure",
      subtitle = "LD-pruned autosomal biallelic SNP PCA of local non-cancer germline WES samples",
      x = xlab12,
      y = ylab
    ) +
    guides(
      shape = guide_legend(order = 1, title.position = "left", nrow = 1),
      color = guide_legend(order = 2, title.position = "left", nrow = 1, override.aes = list(size = 4))
    ) +
    theme_pca(12)

  if (y_col == "PC3") {
    p <- p + labs(x = xlab13, y = ylab13)
  }

  ggsave(file.path(fig_dir, paste0(out_prefix, ".pdf")), p, width = 8.8, height = 6.4, device = cairo_pdf)
  ggsave(file.path(fig_dir, paste0(out_prefix, ".png")), p, width = 8.8, height = 6.4, dpi = 600)
}

make_pca(
  y_col = "PC2",
  ylab = ylab12,
  out_prefix = "Fig_LocalWES_01_PCA_PC1_PC2_population_structure"
)

make_pca(
  y_col = "PC3",
  ylab = ylab13,
  out_prefix = "Fig_LocalWES_01B_PCA_PC1_PC3_population_structure"
)

report <- file.path(status, "78B2_fix_local_germline_PCA_remove_breast_cancer_legend_status.txt")
sink(report)
cat("78B2 fixed local germline PCA figures\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Input table:", pca_table, "\n")
cat("Samples plotted:", nrow(plot_df), "\n")
cat("Groups plotted:\n")
print(table(plot_df$group))
cat("\nSex counts:\n")
print(table(plot_df$sex))
cat("\nFixes:\n")
cat("- Removed unused Breast cancer group from PCA legend.\n")
cat("- Kept only Hadza and Tanzanian control groups.\n")
cat("- Replaced subtitle to avoid unnecessary breast-cancer wording.\n")
cat("- Overwrote Fig_LocalWES_01 and Fig_LocalWES_01B after creating .bak backups.\n")
sink()
RS

echo
echo "=== FIXED PCA FIGURES ==="
find "$FIG_DIR" -maxdepth 1 -type f \( \
  -name "Fig_LocalWES_01_PCA_PC1_PC2_population_structure.*" -o \
  -name "Fig_LocalWES_01B_PCA_PC1_PC3_population_structure.*" \
\) -printf "%f\t%k KB\n" | sort

echo
echo "=== STATUS ==="
cat "$STATUS/78B2_fix_local_germline_PCA_remove_breast_cancer_legend_status.txt"

echo
echo "=== DONE 78B2 ==="
date
