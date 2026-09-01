#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(patchwork)
  library(stringr)
})

# =========================
# INPUT / OUTPUT
# =========================
infile <- file.path(Sys.getenv("PAPER_RESULTS_ROOT"), "05_local_germline_WES", "01_tables", "local_wes_pca_with_metadata.tsv")
outdir <- file.path(Sys.getenv("PAPER_RESULTS_ROOT"), "publication_figures", "local_germline_WES")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

out_png <- file.path(outdir, "Fig_LocalWES_01_PANEL_PCA_PC1_PC2_PC3_population_structure.png")
out_pdf <- file.path(outdir, "Fig_LocalWES_01_PANEL_PCA_PC1_PC2_PC3_population_structure.pdf")

# =========================
# HELPERS
# =========================
pick_col <- function(df, candidates, required = TRUE) {
  hits <- candidates[candidates %in% colnames(df)]
  if (length(hits) > 0) return(hits[1])
  if (required) {
    stop(
      "Could not find required column. Tried: ",
      paste(candidates, collapse = ", "),
      "\nAvailable columns:\n",
      paste(colnames(df), collapse = ", ")
    )
  }
  return(NULL)
}

# =========================
# READ DATA
# =========================
df <- read_tsv(infile, show_col_types = FALSE)

# robust column detection
col_pc1   <- pick_col(df, c("PC1", "pc1"))
col_pc2   <- pick_col(df, c("PC2", "pc2"))
col_pc3   <- pick_col(df, c("PC3", "pc3"))
col_group <- pick_col(df, c("group_label", "group", "origin_label"))
col_sex   <- pick_col(df, c("sex_label", "sex_from_id", "sex"))

# standardize
plot_df <- df %>%
  transmute(
    sample_id = if ("sample_id" %in% colnames(df)) sample_id else row_number(),
    PC1 = .data[[col_pc1]],
    PC2 = .data[[col_pc2]],
    PC3 = .data[[col_pc3]],
    group_raw = as.character(.data[[col_group]]),
    sex_raw   = as.character(.data[[col_sex]])
  ) %>%
  mutate(
    group_label = case_when(
      str_detect(tolower(group_raw), "hadza") ~ "Hadza",
      str_detect(tolower(group_raw), "control") ~ "Tanzanian control",
      TRUE ~ group_raw
    ),
    sex_label = case_when(
      str_detect(tolower(sex_raw), "^f") ~ "Female",
      str_detect(tolower(sex_raw), "^m") ~ "Male",
      TRUE ~ sex_raw
    )
  ) %>%
  filter(group_label %in% c("Hadza", "Tanzanian control")) %>%
  mutate(
    group_label = factor(group_label, levels = c("Hadza", "Tanzanian control")),
    sex_label   = factor(sex_label, levels = c("Female", "Male"))
  )

# =========================
# % variance explained from eigenval file (optional)
# If you want fixed labels as before, leave these numbers.
# =========================
xlab_pc1 <- "PC1 (6.9%)"
ylab_pc2 <- "PC2 (6.5%)"
ylab_pc3 <- "PC3"

# =========================
# THEME
# =========================
base_theme <- theme_classic(base_size = 16) +
  theme(
    plot.title = element_text(face = "bold", size = 22),
    plot.subtitle = element_text(size = 14),
    axis.title = element_text(face = "bold", size = 18),
    axis.text = element_text(size = 14, colour = "black"),
    legend.title = element_text(face = "bold", size = 14),
    legend.text = element_text(size = 13),
    legend.position = "top",
    legend.box = "horizontal",
    strip.text = element_text(face = "bold", size = 14),
    plot.margin = margin(8, 8, 8, 8)
  )

# =========================
# COMMON SCALES
# =========================
col_scale <- scale_color_manual(
  name = "Group",
  values = c("Hadza" = "#18A77A", "Tanzanian control" = "#1F85C7")
)

fill_scale <- scale_fill_manual(
  name = "Group",
  values = c("Hadza" = "#18A77A", "Tanzanian control" = "#1F85C7")
)

shape_scale <- scale_shape_manual(
  name = "Sex",
  values = c("Female" = 16, "Male" = 17)
)

# =========================
# PANEL A: PC1 vs PC2
# =========================
p1 <- ggplot(plot_df, aes(x = PC1, y = PC2)) +
  stat_ellipse(
    aes(color = group_label),
    linewidth = 1.2,
    type = "norm",
    level = 0.80,
    show.legend = FALSE
  ) +
  geom_point(
    aes(color = group_label, shape = sex_label),
    size = 4.2,
    alpha = 0.95
  ) +
  col_scale +
  fill_scale +
  shape_scale +
  labs(
    title = "Local germline WES population structure",
    subtitle = "LD-pruned autosomal biallelic SNP PCA of local non-cancer germline WES samples",
    x = xlab_pc1,
    y = ylab_pc2,
    tag = "A"
  ) +
  base_theme +
  theme(
    plot.tag = element_text(face = "bold", size = 20),
    legend.position = "top"
  )

# =========================
# PANEL B: PC1 vs PC3
# =========================
p2 <- ggplot(plot_df, aes(x = PC1, y = PC3)) +
  stat_ellipse(
    aes(color = group_label),
    linewidth = 1.2,
    type = "norm",
    level = 0.80,
    show.legend = FALSE
  ) +
  geom_point(
    aes(color = group_label, shape = sex_label),
    size = 4.2,
    alpha = 0.95
  ) +
  col_scale +
  fill_scale +
  shape_scale +
  labs(
    title = "Local germline WES population structure",
    subtitle = "LD-pruned autosomal biallelic SNP PCA of local non-cancer germline WES samples",
    x = xlab_pc1,
    y = ylab_pc3,
    tag = "B"
  ) +
  base_theme +
  theme(
    plot.tag = element_text(face = "bold", size = 20),
    legend.position = "top"
  )

# =========================
# COMBINE PANEL
# =========================
panel_plot <- (p1 | p2) +
  plot_layout(guides = "collect") &
  theme(
    legend.position = "top",
    legend.box = "horizontal"
  )

# save
ggsave(out_png, panel_plot, width = 16, height = 7.5, dpi = 300, bg = "white")
ggsave(out_pdf, panel_plot, width = 16, height = 7.5, device = cairo_pdf, bg = "white")

cat("Saved panel figure:\n")
cat(out_png, "\n")
cat(out_pdf, "\n")
