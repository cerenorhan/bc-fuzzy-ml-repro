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

DESEQ_FILE="$PROJECT/results/EA_BC_AI_MultiOmics/rnaseq_main/GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2/DESeq2/annotated/GSE142258_DESeq2_late_vs_early_results.annotated.tsv"

RNA_OUT="$PAPER/04_rnaseq_GSE142258"
RNA_TABLES="$RNA_OUT/01_tables"
RNA_GSEA="$RNA_OUT/02_ranked_pathway_enrichment_noinstall"
RNA_FIGS="$RNA_OUT/03_figures"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$RNA_TABLES" "$RNA_GSEA" "$RNA_FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/74C_paper_results_04A_rnaseq_GSE142258_ranked_pathway_noinstall_fullrank_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 04A-C: RNA-seq full-rank DESeq2 + no-install ranked pathway enrichment ==="
date

echo
echo "=== 1. Input full DESeq2 file ==="

if [[ ! -s "$DESEQ_FILE" ]]; then
  echo "ERROR: Full annotated DESeq2 file not found:"
  echo "$DESEQ_FILE"
  echo
  echo "Available DESeq2 TSV files:"
  find "$PROJECT/results/EA_BC_AI_MultiOmics/rnaseq_main/GSE142258_Trimmomatic_TopHat_featureCounts_DESeq2/DESeq2" \
    -type f -iname "*DESeq2*.tsv" | sort
  exit 1
fi

echo "Selected full DESeq2 file:"
echo "$DESEQ_FILE"
ls -lh "$DESEQ_FILE"

echo
echo "Line count:"
wc -l "$DESEQ_FILE"

echo
echo "Columns:"
head -1 "$DESEQ_FILE" | tr '\t' '\n' | nl -ba

cp -av "$DESEQ_FILE" "$RNA_TABLES/"

echo
echo "=== 2. Check required R packages ==="

micromamba run -n rnaseq Rscript - <<'RS'
pkgs <- c("msigdbr","dplyr","readr","tidyr","stringr","ggplot2","forcats","purrr")
print(sapply(pkgs, requireNamespace, quietly=TRUE))
missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]
if(length(missing) > 0){
  stop("Missing required packages: ", paste(missing, collapse=", "))
}
RS

echo
echo "=== 3. Run ranked pathway enrichment ==="

export DESEQ_FILE RNA_TABLES RNA_GSEA RNA_FIGS STATUS

micromamba run -n rnaseq Rscript - <<'RS'
suppressPackageStartupMessages({
  library(msigdbr)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(forcats)
  library(purrr)
})

deseq_file <- Sys.getenv("DESEQ_FILE")
rna_tables <- Sys.getenv("RNA_TABLES")
rna_gsea <- Sys.getenv("RNA_GSEA")
rna_figs <- Sys.getenv("RNA_FIGS")
status <- Sys.getenv("STATUS")

dir.create(rna_tables, recursive = TRUE, showWarnings = FALSE)
dir.create(rna_gsea, recursive = TRUE, showWarnings = FALSE)
dir.create(rna_figs, recursive = TRUE, showWarnings = FALSE)

df <- read_tsv(deseq_file, show_col_types = FALSE)

write_tsv(
  tibble(column_name = names(df)),
  file.path(rna_tables, "GSE142258_DESeq2_input_columns_detected_fullrank.tsv")
)

required <- c("Geneid", "Geneid_no_version", "gene_name", "gene_type", "log2FoldChange", "stat", "pvalue", "padj")
missing <- setdiff(required, names(df))
if (length(missing) > 0) {
  stop("Missing required columns: ", paste(missing, collapse = ", "))
}

clean <- df %>%
  mutate(
    Geneid = as.character(Geneid),
    Geneid_no_version = as.character(Geneid_no_version),
    gene_symbol = as.character(gene_name),
    gene_symbol = ifelse(is.na(gene_symbol) | gene_symbol == "" | gene_symbol == ".", NA, gene_symbol),
    log2FoldChange = as.numeric(log2FoldChange),
    stat = as.numeric(stat),
    pvalue = as.numeric(pvalue),
    padj = as.numeric(padj),
    rank_metric = case_when(
      !is.na(stat) ~ stat,
      !is.na(log2FoldChange) & !is.na(pvalue) & pvalue > 0 ~ sign(log2FoldChange) * -log10(pvalue),
      TRUE ~ NA_real_
    ),
    direction_DE = case_when(
      !is.na(padj) & padj < 0.05 & log2FoldChange >= 1 ~ "up_padj05_lfc1",
      !is.na(padj) & padj < 0.05 & log2FoldChange <= -1 ~ "down_padj05_lfc1",
      !is.na(padj) & padj < 0.05 ~ "padj05_lfc_lt1",
      TRUE ~ "not_significant"
    )
  ) %>%
  filter(!is.na(gene_symbol), !is.na(rank_metric))

rank_df <- clean %>%
  group_by(gene_symbol) %>%
  arrange(desc(abs(rank_metric)), .by_group = TRUE) %>%
  slice(1) %>%
  ungroup() %>%
  arrange(desc(rank_metric))

if (nrow(rank_df) < 1000) {
  stop("Ranked gene list too small: ", nrow(rank_df), ". Wrong input file likely selected.")
}

ranks <- rank_df$rank_metric
names(ranks) <- rank_df$gene_symbol
ranks <- sort(ranks, decreasing = TRUE)

write_tsv(clean, file.path(rna_tables, "GSE142258_DESeq2_cleaned_for_ranked_enrichment_fullrank.tsv"))

write_tsv(
  tibble(gene_symbol = names(ranks), rank_metric = as.numeric(ranks)),
  file.path(rna_tables, "GSE142258_DESeq2_ranked_gene_list_fullrank.rnk"),
  col_names = FALSE
)

sig_all <- clean %>% filter(!is.na(padj), padj < 0.05)
sig_lfc1 <- clean %>% filter(!is.na(padj), padj < 0.05, abs(log2FoldChange) >= 1)
sig_up <- sig_lfc1 %>% filter(log2FoldChange > 0)
sig_down <- sig_lfc1 %>% filter(log2FoldChange < 0)

write_tsv(sig_all, file.path(rna_tables, "GSE142258_DESeq2_significant_padj05_all_fullrank.tsv"))
write_tsv(sig_lfc1, file.path(rna_tables, "GSE142258_DESeq2_significant_padj05_absLFC1_fullrank.tsv"))
write_tsv(sig_up, file.path(rna_tables, "GSE142258_DESeq2_significant_UP_padj05_absLFC1_fullrank.tsv"))
write_tsv(sig_down, file.path(rna_tables, "GSE142258_DESeq2_significant_DOWN_padj05_absLFC1_fullrank.tsv"))

summary_counts <- tibble(
  metric = c(
    "input_rows",
    "rows_with_gene_symbol_and_rank",
    "unique_ranked_genes",
    "padj05_genes",
    "padj05_absLFC1_genes",
    "up_padj05_absLFC1_genes",
    "down_padj05_absLFC1_genes"
  ),
  value = c(
    nrow(df),
    nrow(clean),
    length(ranks),
    nrow(sig_all),
    nrow(sig_lfc1),
    nrow(sig_up),
    nrow(sig_down)
  )
)

write_tsv(summary_counts, file.path(rna_tables, "GSE142258_DESeq2_summary_counts_fullrank.tsv"))

message("Loading MSigDB through msigdbr...")
msig <- msigdbr(species = "Homo sapiens")

cat_col <- if ("gs_collection" %in% names(msig)) "gs_collection" else "gs_cat"
subcat_col <- if ("gs_subcollection" %in% names(msig)) "gs_subcollection" else "gs_subcat"
gene_col <- if ("gene_symbol" %in% names(msig)) "gene_symbol" else "human_gene_symbol"
desc_col <- if ("gs_description" %in% names(msig)) "gs_description" else NA_character_

msig2 <- msig %>%
  transmute(
    database_raw_collection = .data[[cat_col]],
    database_raw_subcollection = .data[[subcat_col]],
    gs_name = .data[["gs_name"]],
    gs_description = if (!is.na(desc_col)) .data[[desc_col]] else ".",
    gene = .data[[gene_col]]
  ) %>%
  filter(!is.na(gene), gene != "")

sets <- bind_rows(
  msig2 %>%
    filter(database_raw_collection == "H") %>%
    mutate(database = "HALLMARK"),
  msig2 %>%
    filter(str_detect(database_raw_subcollection, "REACTOME") | str_detect(gs_name, "^REACTOME_")) %>%
    mutate(database = "REACTOME"),
  msig2 %>%
    filter(str_detect(database_raw_subcollection, "KEGG") | str_detect(gs_name, "^KEGG_")) %>%
    mutate(database = "KEGG"),
  msig2 %>%
    filter(
      database_raw_collection == "C5" &
      (
        str_detect(database_raw_subcollection, "GO:BP") |
        str_detect(database_raw_subcollection, "BP") |
        str_detect(gs_name, "^GOBP_")
      )
    ) %>%
    mutate(database = "GO_BP")
) %>%
  distinct(database, gs_name, gs_description, gene) %>%
  filter(gene %in% names(ranks))

write_tsv(sets, file.path(rna_gsea, "GSE142258_RNA_msigdbr_gene_sets_used_fullrank.tsv"))

set_summary <- sets %>%
  count(database, gs_name, name = "gene_set_size_in_ranked_genes") %>%
  group_by(database) %>%
  summarise(
    n_gene_sets = n(),
    median_size = median(gene_set_size_in_ranked_genes),
    min_size = min(gene_set_size_in_ranked_genes),
    max_size = max(gene_set_size_in_ranked_genes),
    .groups = "drop"
  )

write_tsv(set_summary, file.path(rna_gsea, "GSE142258_RNA_msigdbr_gene_set_summary_fullrank.tsv"))

calc_es <- function(stats, geneset, p = 1) {
  stats <- sort(stats, decreasing = TRUE)
  hits <- names(stats) %in% geneset
  Nh <- sum(hits)
  N <- length(stats)
  if (Nh == 0 || Nh == N) return(NA_real_)

  norm_hit <- sum(abs(stats[hits])^p)
  running <- cumsum(ifelse(hits, abs(stats)^p / norm_hit, -1 / (N - Nh)))

  max_es <- max(running)
  min_es <- min(running)
  if (abs(max_es) >= abs(min_es)) max_es else min_es
}

leading_edge_genes <- function(stats, geneset, es) {
  stats <- sort(stats, decreasing = TRUE)
  hits <- names(stats) %in% geneset
  Nh <- sum(hits)
  N <- length(stats)
  if (Nh == 0 || Nh == N) return(".")

  norm_hit <- sum(abs(stats[hits]))
  running <- cumsum(ifelse(hits, abs(stats) / norm_hit, -1 / (N - Nh)))
  peak <- if (es >= 0) which.max(running) else which.min(running)

  le <- names(stats)[seq_len(peak)][hits[seq_len(peak)]]
  if (length(le) == 0) "." else paste(le, collapse = ",")
}

run_ranked_enrichment <- function(database_name, minSize = 10, maxSize = 500) {
  dbsets <- sets %>%
    filter(database == database_name) %>%
    group_by(database, gs_name, gs_description) %>%
    summarise(genes = list(sort(unique(gene))), .groups = "drop") %>%
    mutate(size = map_int(genes, ~ length(intersect(.x, names(ranks))))) %>%
    filter(size >= minSize, size <= maxSize)

  if (nrow(dbsets) == 0) {
    return(tibble())
  }

  bg <- as.numeric(ranks)

  res <- dbsets %>%
    rowwise() %>%
    mutate(
      ES = calc_es(ranks, genes),
      leadingEdge = leading_edge_genes(ranks, genes, ES),
      mean_rank_metric = mean(as.numeric(ranks[intersect(genes, names(ranks))]), na.rm = TRUE),
      median_rank_metric = median(as.numeric(ranks[intersect(genes, names(ranks))]), na.rm = TRUE),
      rank_shift_z = {
        x <- as.numeric(ranks[intersect(genes, names(ranks))])
        ifelse(sd(bg, na.rm = TRUE) > 0, (mean(x, na.rm = TRUE) - mean(bg, na.rm = TRUE)) / sd(bg, na.rm = TRUE), NA_real_)
      },
      p_up = {
        x <- as.numeric(ranks[intersect(genes, names(ranks))])
        suppressWarnings(wilcox.test(x, bg, alternative = "greater", exact = FALSE)$p.value)
      },
      p_down = {
        x <- as.numeric(ranks[intersect(genes, names(ranks))])
        suppressWarnings(wilcox.test(x, bg, alternative = "less", exact = FALSE)$p.value)
      },
      pvalue = min(p_up, p_down, na.rm = TRUE),
      direction = ifelse(p_up <= p_down, "up_in_late", "down_in_late")
    ) %>%
    ungroup() %>%
    group_by(database) %>%
    mutate(padj = p.adjust(pvalue, method = "BH")) %>%
    ungroup() %>%
    select(
      database, gs_name, gs_description, size,
      ES, mean_rank_metric, median_rank_metric, rank_shift_z,
      p_up, p_down, pvalue, padj, direction, leadingEdge
    ) %>%
    arrange(padj, pvalue, desc(abs(rank_shift_z)))

  res
}

enrich_all <- bind_rows(
  run_ranked_enrichment("HALLMARK"),
  run_ranked_enrichment("REACTOME"),
  run_ranked_enrichment("KEGG"),
  run_ranked_enrichment("GO_BP")
)

if (nrow(enrich_all) == 0) {
  stop("Ranked enrichment returned zero rows. Check gene symbols/MSigDB overlap.")
}

write_tsv(enrich_all, file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_enrichment_all_results.tsv"))

enrich_sig <- enrich_all %>%
  filter(!is.na(padj), padj <= 0.25) %>%
  arrange(padj, pvalue, desc(abs(rank_shift_z)))

write_tsv(enrich_sig, file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_enrichment_FDR025.tsv"))

enrich_top <- enrich_all %>%
  group_by(database, direction) %>%
  arrange(padj, pvalue, desc(abs(rank_shift_z)), .by_group = TRUE) %>%
  slice_head(n = 25) %>%
  ungroup()

write_tsv(enrich_top, file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_enrichment_top25_per_database_direction.tsv"))

leading_long <- enrich_all %>%
  filter(leadingEdge != "." & leadingEdge != "") %>%
  separate_rows(leadingEdge, sep = ",") %>%
  rename(gene_symbol = leadingEdge) %>%
  left_join(
    rank_df %>% select(gene_symbol, log2FoldChange, stat, pvalue, padj, rank_metric, direction_DE),
    by = "gene_symbol"
  )

write_tsv(leading_long, file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_leading_edge_genes_long.tsv"))

enrich_summary <- enrich_all %>%
  group_by(database) %>%
  summarise(
    n_terms_tested = n(),
    n_FDR025 = sum(padj <= 0.25, na.rm = TRUE),
    n_FDR010 = sum(padj <= 0.10, na.rm = TRUE),
    best_padj = min(padj, na.rm = TRUE),
    top_term = gs_name[which.min(padj)],
    .groups = "drop"
  )

write_tsv(enrich_summary, file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_enrichment_summary_by_database.tsv"))

# Volcano figure
vol <- clean %>%
  mutate(
    neglog10padj = ifelse(!is.na(padj) & padj > 0, -log10(padj), NA_real_),
    sig_class = case_when(
      !is.na(padj) & padj < 0.05 & log2FoldChange >= 1 ~ "Up",
      !is.na(padj) & padj < 0.05 & log2FoldChange <= -1 ~ "Down",
      TRUE ~ "NS"
    )
  )

p <- ggplot(vol, aes(x = log2FoldChange, y = neglog10padj)) +
  geom_point(aes(shape = sig_class), alpha = 0.45, size = 1.2) +
  theme_bw(base_size = 10) +
  labs(
    title = "GSE142258 RNA-seq DESeq2: late vs early",
    x = "log2 fold change",
    y = "-log10 adjusted p-value"
  )

ggsave(file.path(rna_figs, "GSE142258_DESeq2_volcano_fullrank.png"), p, width = 7, height = 5, dpi = 300)

for (db in unique(enrich_all$database)) {
  dat <- enrich_all %>%
    filter(database == db) %>%
    arrange(padj, pvalue, desc(abs(rank_shift_z))) %>%
    slice_head(n = 20) %>%
    mutate(
      term_short = gs_name %>%
        str_replace("^HALLMARK_", "") %>%
        str_replace("^REACTOME_", "") %>%
        str_replace("^KEGG_", "") %>%
        str_replace("^GOBP_", "") %>%
        str_replace_all("_", " ") %>%
        str_to_title(),
      signed_score = ifelse(direction == "up_in_late", -log10(padj + 1e-300), log10(padj + 1e-300))
    )

  if (nrow(dat) > 0) {
    p <- ggplot(dat, aes(x = signed_score, y = fct_reorder(term_short, signed_score))) +
      geom_col() +
      theme_bw(base_size = 10) +
      labs(
        title = paste0("RNA ranked pathway enrichment: ", db),
        x = "signed -log10 FDR; positive=up in late",
        y = NULL
      )

    ggsave(
      file.path(rna_figs, paste0("GSE142258_RNA_ranked_pathway_top_", db, ".png")),
      p,
      width = 8.5,
      height = max(4.5, nrow(dat) * 0.28),
      dpi = 300
    )
  }
}

report <- file.path(status, "GSE142258_stage_04A_C_RNA_ranked_pathway_noinstall_fullrank_status.txt")

sink(report)
cat("GSE142258 Stage 04A-C RNA-seq full-rank DESeq2 + no-install ranked pathway enrichment\n")
cat("Generated:", as.character(Sys.time()), "\n")
cat("Input:", deseq_file, "\n")
cat("Gene symbol column: gene_name\n")
cat("Note: This is Wilcoxon rank-based pathway enrichment using the full DESeq2 ranked list; fgsea package was not used.\n\n")
cat("Summary counts\n")
print(summary_counts)
cat("\nGene set summary\n")
print(set_summary)
cat("\nRanked pathway enrichment summary\n")
print(enrich_summary)
cat("\nTop ranked pathway enrichment results\n")
print(enrich_top %>% select(database, gs_name, size, rank_shift_z, pvalue, padj, direction, leadingEdge) %>% head(120))
cat("\nFDR <= 0.25 results\n")
print(enrich_sig %>% select(database, gs_name, size, rank_shift_z, pvalue, padj, direction, leadingEdge) %>% head(150))
sink()

cat("summary_counts", file.path(rna_tables, "GSE142258_DESeq2_summary_counts_fullrank.tsv"), "\n")
cat("ranked_enrichment_summary", file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_enrichment_summary_by_database.tsv"), "\n")
cat("ranked_enrichment_all", file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_enrichment_all_results.tsv"), "\n")
cat("ranked_enrichment_FDR025", file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_enrichment_FDR025.tsv"), "\n")
cat("ranked_enrichment_top25", file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_enrichment_top25_per_database_direction.tsv"), "\n")
cat("leading_edge", file.path(rna_gsea, "GSE142258_RNA_ranked_pathway_leading_edge_genes_long.tsv"), "\n")
cat("report", report, "\n")
RS

echo
echo "=== RNA DESeq2 SUMMARY COUNTS ==="
column -t -s $'\t' "$RNA_TABLES/GSE142258_DESeq2_summary_counts_fullrank.tsv"

echo
echo "=== RNA RANKED PATHWAY SUMMARY BY DATABASE ==="
column -t -s $'\t' "$RNA_GSEA/GSE142258_RNA_ranked_pathway_enrichment_summary_by_database.tsv"

echo
echo "=== RNA RANKED PATHWAY TOP RESULTS ==="
column -t -s $'\t' "$RNA_GSEA/GSE142258_RNA_ranked_pathway_enrichment_top25_per_database_direction.tsv" | head -100

echo
echo "=== RNA RANKED PATHWAY FDR <= 0.25 ==="
if [[ -s "$RNA_GSEA/GSE142258_RNA_ranked_pathway_enrichment_FDR025.tsv" ]]; then
  column -t -s $'\t' "$RNA_GSEA/GSE142258_RNA_ranked_pathway_enrichment_FDR025.tsv" | head -120
else
  echo "No RNA ranked pathway terms at FDR <= 0.25."
fi

echo
echo "=== OUTPUT FILES ==="
find "$RNA_OUT" -maxdepth 3 -type f -printf "%P\t%k KB\n" | sort

echo
echo "=== DONE: STAGE 04A-C ==="
date
