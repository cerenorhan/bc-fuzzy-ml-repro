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

RNA_TERMS="$PAPER/04_rnaseq_GSE142258/04_interpretable_ranked_pathway_summary/GSE142258_RNA_ranked_pathway_primary_interpretable_FDR025.tsv"

OUT="$PAPER/publication_figures/GSE142258_RNA_polished"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUT" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/77A3_publication_final_RNA_selected_pathway_dotplot_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== 77A3: FINAL SELECTED RNA PATHWAY DOTPLOT ==="
date

if [[ ! -s "$RNA_TERMS" ]]; then
  echo "ERROR: Missing input:"
  echo "$RNA_TERMS"
  exit 1
fi

export RNA_TERMS OUT STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(stringr)
  library(forcats)
  library(tibble)
})

terms_file <- Sys.getenv("RNA_TERMS")
outdir <- Sys.getenv("OUT")
statusdir <- Sys.getenv("STATUS")

dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

theme_pub <- function(base_size = 11) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 3),
      plot.subtitle = element_text(size = base_size),
      axis.title = element_text(face = "bold"),
      axis.text = element_text(color = "grey20"),
      axis.line = element_line(linewidth = 0.35, color = "grey20"),
      legend.title = element_text(face = "bold"),
      legend.position = "right",
      strip.background = element_rect(fill = "grey92", color = NA),
      strip.text = element_text(face = "bold", size = base_size + 1),
      panel.spacing.y = unit(0.55, "lines")
    )
}

nice_axis <- function(x) {
  recode(
    x,
    "translation_ribosome_RNA" = "Translation / ribosome / rRNA",
    "immune_antigen_T_NK" = "Immune antigen / T-cell / NK",
    "cell_cycle_DNA_repair" = "Cell cycle / DNA repair",
    "development_Notch_TGF_WNT" = "Notch / TGF-beta / WNT",
    "hormone_luminal" = "Hormone / luminal signaling",
    "metabolism_hypoxia_lipid" = "Metabolism / hypoxia / lipid",
    "EMT_invasion_angiogenesis" = "EMT / invasion / angiogenesis",
    "EMT_invasion_angiogenesis;immune_antigen_T_NK" = "EMT + immune migration",
    "immune_antigen_T_NK;metabolism_hypoxia_lipid" = "Immune + metabolic",
    "cell_cycle_DNA_repair;immune_antigen_T_NK" = "Cell cycle + immune",
    .default = x
  )
}

axis_cols <- c(
  "Translation / ribosome / rRNA" = "#1F77B4",
  "Immune antigen / T-cell / NK" = "#009E73",
  "Cell cycle / DNA repair" = "#D55E00",
  "Notch / TGF-beta / WNT" = "#CC79A7",
  "Hormone / luminal signaling" = "#7E57C2",
  "Metabolism / hypoxia / lipid" = "#E69F00",
  "EMT / invasion / angiogenesis" = "#00A6D6",
  "EMT + immune migration" = "#00BFC4",
  "Immune + metabolic" = "#66A61E",
  "Cell cycle + immune" = "#A6761D"
)

terms <- read_tsv(terms_file, show_col_types = FALSE) %>%
  mutate(
    padj = as.numeric(padj),
    pvalue = as.numeric(pvalue),
    size = as.numeric(size),
    neglog10_fdr = -log10(padj + 1e-300),
    term_upper = toupper(gs_name),
    axis_label = nice_axis(biology_axis),
    direction_label = ifelse(direction == "up_in_late", "Up in late", "Down in late")
  ) %>%
  filter(reporting_tier == "interpretable_FDR025") %>%
  filter(biology_axis != "other") %>%
  filter(!str_detect(term_upper, "MEDICUS")) %>%
  filter(!is.na(padj), !is.na(size))

# Temsilci pathway paneli:
# Her satır, ilgili eksen ve yön içinde mevcut en anlamlı eşleşen terimi seçer.
targets <- tribble(
  ~priority, ~direction,      ~axis_match,                    ~pattern,                                                                       ~display_label,
   1,        "down_in_late",  "translation_ribosome_RNA",     "EUKARYOTIC_TRANSLATION_ELONGATION",                                            "Eukaryotic translation elongation",
   2,        "down_in_late",  "translation_ribosome_RNA",     "^KEGG_RIBOSOME$|RIBOSOME_ASSEMBLY",                                            "Ribosome / ribosome assembly",
   3,        "down_in_late",  "translation_ribosome_RNA",     "RRNA_PROCESSING",                                                              "rRNA processing",

   4,        "down_in_late",  "immune_antigen_T_NK",          "ALLOGRAFT_REJECTION",                                                          "Allograft / immune rejection signature",
   5,        "down_in_late",  "immune_antigen_T_NK",          "INTERFERON_GAMMA_RESPONSE|INTERFERON_ALPHA_RESPONSE|INTERFERON_SIGNALING",     "Interferon response",
   6,        "down_in_late",  "immune_antigen_T_NK",          "ANTIGEN_PROCESSING|CLASS_I_MHC|ANTIGEN_PRESENTATION",                          "Antigen processing and presentation",
   7,        "down_in_late",  "immune_antigen_T_NK",          "T_CELL_ACTIVATION|TCR_SIGNALING|CD28",                                         "T-cell receptor / activation",

   8,        "down_in_late",  "cell_cycle_DNA_repair",        "HALLMARK_E2F_TARGETS",                                                         "E2F targets",
   9,        "down_in_late",  "cell_cycle_DNA_repair",        "HALLMARK_G2M_CHECKPOINT|G2_M_CHECKPOINT",                                      "G2/M checkpoint",
  10,        "down_in_late",  "cell_cycle_DNA_repair",        "DNA_REPAIR|DOUBLE_STRAND_BREAK_REPAIR",                                        "DNA repair / double-strand break repair",

  11,        "down_in_late",  "development_Notch_TGF_WNT",    "NOTCH_SIGNALING|SIGNALING_BY_NOTCH",                                           "Notch signaling",
  12,        "down_in_late",  "development_Notch_TGF_WNT",    "TGF_BETA_SIGNALING|SIGNALING_BY_TGFB",                                         "TGF-beta signaling",
  13,        "down_in_late",  "development_Notch_TGF_WNT",    "TCF_DEPENDENT_SIGNALING_IN_RESPONSE_TO_WNT|SIGNALING_BY_WNT",                  "WNT / TCF signaling",

  14,        "down_in_late",  "hormone_luminal",              "ESTROGEN_DEPENDENT_GENE_EXPRESSION|ESTROGEN_RESPONSE|ANDROGEN_RESPONSE",       "Hormone receptor signaling",

  15,        "up_in_late",    "EMT_invasion_angiogenesis",    "HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION",                                   "Epithelial-mesenchymal transition",
  16,        "up_in_late",    "EMT_invasion_angiogenesis",    "HALLMARK_ANGIOGENESIS",                                                        "Angiogenesis",
  17,        "up_in_late",    "EMT_invasion_angiogenesis",    "HALLMARK_COAGULATION",                                                         "Coagulation",
  18,        "up_in_late",    "EMT_invasion_angiogenesis",    "EXTRACELLULAR_MATRIX_ORGANIZATION|COLLAGEN_FORMATION|COLLAGEN_BIOSYNTHESIS",   "ECM / collagen remodeling",

  19,        "up_in_late",    "metabolism_hypoxia_lipid",     "HALLMARK_CHOLESTEROL_HOMEOSTASIS",                                             "Cholesterol homeostasis",
  20,        "up_in_late",    "metabolism_hypoxia_lipid",     "HYPOXIA|HYPOXIA_INDUCIBLE_FACTOR",                                             "Hypoxia-related signaling"
)

selected_list <- list()
missing_list <- list()

for (i in seq_len(nrow(targets))) {
  tr <- targets[i, ]

  hit <- terms %>%
    filter(direction == tr$direction) %>%
    filter(str_detect(biology_axis, fixed(tr$axis_match))) %>%
    filter(str_detect(term_upper, regex(tr$pattern, ignore_case = TRUE))) %>%
    arrange(padj, pvalue, desc(size)) %>%
    slice_head(n = 1)

  if (nrow(hit) == 1) {
    hit <- hit %>%
      mutate(
        priority = tr$priority,
        display_label = tr$display_label,
        target_pattern = tr$pattern
      )
    selected_list[[length(selected_list) + 1]] <- hit
  } else {
    missing_list[[length(missing_list) + 1]] <- tr
  }
}

selected <- bind_rows(selected_list) %>%
  arrange(priority) %>%
  distinct(gs_name, .keep_all = TRUE) %>%
  mutate(
    display_label_wrapped = str_wrap(display_label, width = 34),
    direction_label = factor(direction_label, levels = c("Down in late", "Up in late")),
    display_label_wrapped = factor(display_label_wrapped, levels = rev(unique(display_label_wrapped)))
  )

missing <- bind_rows(missing_list)

write_tsv(selected, file.path(outdir, "RNA_selected_representative_pathway_terms_final.tsv"))
if (nrow(missing) > 0) {
  write_tsv(missing, file.path(outdir, "RNA_selected_representative_pathway_terms_missing_patterns.tsv"))
}

# Final main-figure dotplot / lollipop
p <- ggplot(selected, aes(x = neglog10_fdr, y = display_label_wrapped)) +
  geom_segment(
    aes(x = 0, xend = neglog10_fdr, yend = display_label_wrapped, color = axis_label),
    linewidth = 0.75,
    alpha = 0.45
  ) +
  geom_point(
    aes(size = size, fill = axis_label),
    shape = 21,
    color = "grey15",
    stroke = 0.35,
    alpha = 0.95
  ) +
  facet_grid(direction_label ~ ., scales = "free_y", space = "free_y") +
  scale_fill_manual(values = axis_cols, drop = FALSE) +
  scale_color_manual(values = axis_cols, drop = FALSE) +
  scale_size_continuous(range = c(3.2, 8.8)) +
  scale_x_continuous(expand = expansion(mult = c(0.01, 0.08))) +
  labs(
    title = "Representative RNA pathway shifts",
    subtitle = "Selected interpretable ranked pathway terms; full pathway table retained as supplementary output",
    x = expression(-log[10](FDR)),
    y = NULL,
    fill = "Biological axis",
    color = "Biological axis",
    size = "Gene-set size"
  ) +
  theme_pub(10) +
  theme(
    axis.text.y = element_text(size = 9.8),
    legend.position = "right",
    legend.box = "vertical"
  ) +
  guides(
    color = "none",
    fill = guide_legend(override.aes = list(size = 4)),
    size = guide_legend(order = 2)
  )

ggsave(file.path(outdir, "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL.pdf"), p, width = 9.4, height = 6.8)
ggsave(file.path(outdir, "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL.png"), p, width = 9.4, height = 6.8, dpi = 600)

# Daha kompakt legendsiz versiyon; panel birleşiminde işe yarar.
p_compact <- p +
  theme(
    legend.position = "none",
    plot.title = element_text(size = 13),
    plot.subtitle = element_text(size = 10)
  )

ggsave(file.path(outdir, "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL_no_legend.pdf"), p_compact, width = 7.4, height = 6.4)
ggsave(file.path(outdir, "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL_no_legend.png"), p_compact, width = 7.4, height = 6.4, dpi = 600)

report <- file.path(statusdir, "77A3_publication_final_RNA_selected_pathway_dotplot_status.txt")
sink(report)
cat("77A3 final selected representative RNA pathway dotplot\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Input:", terms_file, "\n")
cat("Selected pathway count:", nrow(selected), "\n")
cat("Missing representative pattern count:", nrow(missing), "\n\n")
cat("Selected terms:\n")
print(selected %>% select(priority, display_label, database, gs_name, axis_label, direction_label, size, padj, neglog10_fdr))
if (nrow(missing) > 0) {
  cat("\nMissing patterns:\n")
  print(missing)
}
sink()

cat("selected_terms", file.path(outdir, "RNA_selected_representative_pathway_terms_final.tsv"), "\n")
cat("missing_patterns", ifelse(nrow(missing) > 0, file.path(outdir, "RNA_selected_representative_pathway_terms_missing_patterns.tsv"), "none"), "\n")
cat("figure_pdf", file.path(outdir, "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL.pdf"), "\n")
cat("figure_png", file.path(outdir, "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL.png"), "\n")
cat("figure_no_legend_pdf", file.path(outdir, "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL_no_legend.pdf"), "\n")
cat("figure_no_legend_png", file.path(outdir, "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL_no_legend.png"), "\n")
cat("report", report, "\n")
RS

echo
echo "=== FINAL SELECTED RNA PATHWAY FIGURES ==="
find "$OUT" -maxdepth 1 -type f -name "Fig_RNA_03_selected_representative_pathway_dotplot_FINAL*" -printf "%f\t%k KB\n" | sort

echo
echo "=== SELECTED TERMS ==="
column -t -s $'\t' "$OUT/RNA_selected_representative_pathway_terms_final.tsv" | cut -c1-220 | head -80

echo
echo "=== STATUS ==="
cat "$STATUS/77A3_publication_final_RNA_selected_pathway_dotplot_status.txt" | head -120

echo
echo "=== DONE 77A3 ==="
date
