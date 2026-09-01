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


PROJECT="${PROJECT:-$PROJECT_ROOT}"
PAPER="$PAPER_RESULTS_ROOT"

SOMATIC="$PAPER/03_somatic_wes_PRJNA913947"
GENE_SUMMARY="$SOMATIC/06_unbiased_gene_summary"
OUTDIR="$SOMATIC/07_database_pathway_enrichment"
FIGS="$SOMATIC/02_figures"
STATUS="$PAPER/00_status"
LOGDIR="$PAPER/run_logs"

mkdir -p "$OUTDIR" "$FIGS" "$STATUS" "$LOGDIR"

LOG="$LOGDIR/73B_paper_results_03B_database_somatic_pathway_ORA_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

echo "=== STAGE 03B: DATABASE-DRIVEN SOMATIC PATHWAY ORA ==="
date

BACKGROUND="$GENE_SUMMARY/PRJNA913947_somatic_background_all_annotated_genes.txt"
QUERY_EXT="$GENE_SUMMARY/PRJNA913947_somatic_query_functional_extended_genes.txt"
QUERY_STRICT="$GENE_SUMMARY/PRJNA913947_somatic_query_strict_HIGH_MODERATE_genes.txt"
QUERY_NOCAUTION="$GENE_SUMMARY/PRJNA913947_somatic_query_functional_extended_no_caution_genes.txt"
GENE_TABLE="$GENE_SUMMARY/PRJNA913947_unbiased_functional_gene_summary.tsv"

for f in "$BACKGROUND" "$QUERY_EXT" "$QUERY_STRICT" "$QUERY_NOCAUTION" "$GENE_TABLE"
do
  if [[ ! -s "$f" ]]; then
    echo "ERROR: Missing required input: $f"
    exit 1
  fi
done

echo
echo "Input gene counts:"
echo -n "background_all_annotated_genes: "; wc -l < "$BACKGROUND"
echo -n "query_functional_extended_genes: "; wc -l < "$QUERY_EXT"
echo -n "query_strict_HIGH_MODERATE_genes: "; wc -l < "$QUERY_STRICT"
echo -n "query_functional_extended_no_caution_genes: "; wc -l < "$QUERY_NOCAUTION"

echo
echo "=== 1. Check/install R packages in rnaseq environment ==="

if ! micromamba run -n rnaseq Rscript -e 'pkgs <- c("msigdbr","dplyr","readr","tidyr","stringr","ggplot2","forcats","purrr"); missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1), quietly=TRUE)]; if(length(missing)>0){cat("missing:", paste(missing, collapse=","), "\n"); quit(status=1)}' ; then
  echo "Installing missing R packages into rnaseq env..."
  micromamba install -y -n rnaseq -c conda-forge -c bioconda \
    r-msigdbr r-dplyr r-readr r-tidyr r-stringr r-ggplot2 r-forcats r-purrr
fi

echo
echo "=== 2. Run ORA using MSigDB-derived gene sets ==="

export BACKGROUND QUERY_EXT QUERY_STRICT QUERY_NOCAUTION GENE_TABLE OUTDIR FIGS STATUS

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

background_path <- Sys.getenv("BACKGROUND")
query_ext_path <- Sys.getenv("QUERY_EXT")
query_strict_path <- Sys.getenv("QUERY_STRICT")
query_nocaution_path <- Sys.getenv("QUERY_NOCAUTION")
gene_table_path <- Sys.getenv("GENE_TABLE")
outdir <- Sys.getenv("OUTDIR")
figs <- Sys.getenv("FIGS")
status <- Sys.getenv("STATUS")

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)
dir.create(figs, showWarnings = FALSE, recursive = TRUE)

read_gene_list <- function(path) {
  x <- readLines(path, warn = FALSE)
  x <- unique(trimws(x))
  x <- x[x != "" & x != "." & !is.na(x)]
  sort(x)
}

background <- read_gene_list(background_path)
query_ext <- read_gene_list(query_ext_path)
query_strict <- read_gene_list(query_strict_path)
query_nocaution <- read_gene_list(query_nocaution_path)

gene_table <- read_tsv(gene_table_path, show_col_types = FALSE)

message("Loading MSigDB via msigdbr...")
msig <- msigdbr(species = "Homo sapiens")

write_tsv(
  tibble(column_name = names(msig)),
  file.path(outdir, "msigdbr_columns_detected.tsv")
)

cat_col <- if ("gs_collection" %in% names(msig)) "gs_collection" else "gs_cat"
subcat_col <- if ("gs_subcollection" %in% names(msig)) "gs_subcollection" else "gs_subcat"
gene_col <- if ("gene_symbol" %in% names(msig)) "gene_symbol" else if ("human_gene_symbol" %in% names(msig)) "human_gene_symbol" else NA_character_

if (is.na(gene_col)) {
  stop("Could not detect gene symbol column in msigdbr output.")
}

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

inventory <- msig2 %>%
  count(database_raw_collection, database_raw_subcollection, sort = TRUE)

write_tsv(inventory, file.path(outdir, "msigdbr_collection_inventory.tsv"))

hallmark <- msig2 %>%
  filter(database_raw_collection == "H") %>%
  mutate(database = "HALLMARK")

reactome <- msig2 %>%
  filter(
    str_detect(database_raw_subcollection, "REACTOME") |
      str_detect(gs_name, "^REACTOME_")
  ) %>%
  mutate(database = "REACTOME")

kegg <- msig2 %>%
  filter(
    str_detect(database_raw_subcollection, "KEGG") |
      str_detect(gs_name, "^KEGG_")
  ) %>%
  mutate(database = "KEGG")

go_bp <- msig2 %>%
  filter(
    database_raw_collection == "C5" &
      (
        str_detect(database_raw_subcollection, "GO:BP") |
          str_detect(database_raw_subcollection, "BP") |
          str_detect(gs_name, "^GOBP_")
      )
  ) %>%
  mutate(database = "GO_BP")

sets <- bind_rows(hallmark, reactome, kegg, go_bp) %>%
  distinct(database, gs_name, gs_description, gene) %>%
  filter(gene %in% background)

write_tsv(sets, file.path(outdir, "PRJNA913947_somatic_msigdbr_gene_sets_used.tsv"))

set_summary <- sets %>%
  distinct(database, gs_name, gene) %>%
  count(database, gs_name, name = "gene_set_size_in_background") %>%
  group_by(database) %>%
  summarise(
    n_gene_sets = n(),
    median_gene_set_size = median(gene_set_size_in_background),
    min_gene_set_size = min(gene_set_size_in_background),
    max_gene_set_size = max(gene_set_size_in_background),
    .groups = "drop"
  )

write_tsv(set_summary, file.path(outdir, "PRJNA913947_somatic_msigdbr_gene_set_summary.tsv"))

run_ora <- function(query_genes, universe_genes, sets_df, query_label, min_size = 10, max_size = 500) {
  query_genes <- intersect(unique(query_genes), universe_genes)

  term_list <- sets_df %>%
    group_by(database, gs_name, gs_description) %>%
    summarise(
      term_genes = list(sort(unique(gene))),
      .groups = "drop"
    ) %>%
    mutate(
      term_genes_bg = map(term_genes, ~ intersect(.x, universe_genes)),
      gene_set_size = map_int(term_genes_bg, length)
    ) %>%
    filter(gene_set_size >= min_size, gene_set_size <= max_size)

  M <- length(universe_genes)
  N <- length(query_genes)

  res <- term_list %>%
    mutate(
      overlap_genes_list = map(term_genes_bg, ~ intersect(.x, query_genes)),
      overlap_n = map_int(overlap_genes_list, length),
      query_size = N,
      background_size = M,
      a = overlap_n,
      b = query_size - overlap_n,
      c = gene_set_size - overlap_n,
      d = background_size - a - b - c,
      odds_ratio = pmap_dbl(
        list(a, b, c, d),
        function(a, b, c, d) {
          if (any(c(a, b, c, d) < 0)) return(NA_real_)
          suppressWarnings(fisher.test(matrix(c(a, b, c, d), nrow = 2), alternative = "greater")$estimate)
        }
      ),
      pvalue = pmap_dbl(
        list(a, b, c, d),
        function(a, b, c, d) {
          if (any(c(a, b, c, d) < 0)) return(1)
          fisher.test(matrix(c(a, b, c, d), nrow = 2), alternative = "greater")$p.value
        }
      ),
      query_label = query_label,
      overlap_genes = map_chr(overlap_genes_list, ~ ifelse(length(.x) == 0, ".", paste(.x, collapse = ","))),
      minus_log10_pvalue = ifelse(pvalue > 0, -log10(pvalue), 300)
    ) %>%
    group_by(query_label, database) %>%
    mutate(qvalue_BH_within_database = p.adjust(pvalue, method = "BH")) %>%
    ungroup() %>%
    mutate(qvalue_BH_global = p.adjust(pvalue, method = "BH")) %>%
    select(
      query_label, database, gs_name, gs_description,
      overlap_n, gene_set_size, query_size, background_size,
      odds_ratio, pvalue, qvalue_BH_within_database, qvalue_BH_global,
      minus_log10_pvalue, overlap_genes
    ) %>%
    arrange(query_label, database, qvalue_BH_within_database, pvalue, desc(overlap_n))

  res
}

queries <- list(
  functional_extended = query_ext,
  strict_HIGH_MODERATE = query_strict,
  functional_extended_no_caution_sensitivity = query_nocaution
)

all_ora <- imap_dfr(queries, ~ run_ora(.x, background, sets, .y))

all_out <- file.path(outdir, "PRJNA913947_somatic_database_ORA_all_results.tsv")
write_tsv(all_ora, all_out)

top_results <- all_ora %>%
  group_by(query_label, database) %>%
  arrange(qvalue_BH_within_database, pvalue, desc(overlap_n), .by_group = TRUE) %>%
  slice_head(n = 25) %>%
  ungroup()

top_out <- file.path(outdir, "PRJNA913947_somatic_database_ORA_top25_per_query_database.tsv")
write_tsv(top_results, top_out)

sig_results <- all_ora %>%
  filter(qvalue_BH_within_database <= 0.25, overlap_n >= 2) %>%
  arrange(query_label, database, qvalue_BH_within_database, pvalue, desc(overlap_n))

sig_out <- file.path(outdir, "PRJNA913947_somatic_database_ORA_FDR025.tsv")
write_tsv(sig_results, sig_out)

overlap_long <- all_ora %>%
  filter(overlap_genes != ".") %>%
  separate_rows(overlap_genes, sep = ",") %>%
  rename(gene = overlap_genes) %>%
  left_join(
    gene_table %>%
      select(
        gene,
        n_pairs_any,
        n_variants_any,
        n_pairs_functional_extended,
        n_functional_extended,
        n_pairs_high,
        n_high,
        n_pairs_moderate,
        n_moderate,
        unbiased_somatic_gene_score,
        unbiased_somatic_gene_score_caution_adjusted,
        caution_reason,
        has_caution_flag
      ),
    by = "gene"
  )

overlap_long_out <- file.path(outdir, "PRJNA913947_somatic_database_ORA_overlap_genes_long.tsv")
write_tsv(overlap_long, overlap_long_out)

summary_by_query_db <- all_ora %>%
  group_by(query_label, database) %>%
  summarise(
    n_terms_tested = n(),
    n_terms_FDR_0_25 = sum(qvalue_BH_within_database <= 0.25 & overlap_n >= 2, na.rm = TRUE),
    n_terms_FDR_0_10 = sum(qvalue_BH_within_database <= 0.10 & overlap_n >= 2, na.rm = TRUE),
    top_term = gs_name[order(qvalue_BH_within_database, pvalue)][1],
    top_qvalue = min(qvalue_BH_within_database, na.rm = TRUE),
    .groups = "drop"
  )

summary_out <- file.path(outdir, "PRJNA913947_somatic_database_ORA_summary_by_query_database.tsv")
write_tsv(summary_by_query_db, summary_out)

plot_top <- function(res, query_lab, db_lab, outfile) {
  dat <- res %>%
    filter(query_label == query_lab, database == db_lab, overlap_n >= 2) %>%
    arrange(qvalue_BH_within_database, pvalue, desc(overlap_n)) %>%
    slice_head(n = 20) %>%
    mutate(
      term_short = gs_name %>%
        str_replace("^HALLMARK_", "") %>%
        str_replace("^REACTOME_", "") %>%
        str_replace("^KEGG_", "") %>%
        str_replace("^GOBP_", "") %>%
        str_replace_all("_", " ") %>%
        str_to_title()
    )

  if (nrow(dat) == 0) return(invisible(NULL))

  p <- ggplot(dat, aes(x = minus_log10_pvalue, y = fct_reorder(term_short, minus_log10_pvalue))) +
    geom_col() +
    labs(
      x = "-log10 p-value",
      y = NULL,
      title = paste0("Somatic ORA: ", query_lab, " / ", db_lab)
    ) +
    theme_bw(base_size = 10)

  ggsave(outfile, p, width = 8.5, height = max(4.5, nrow(dat) * 0.28), dpi = 300)
}

for (q in unique(all_ora$query_label)) {
  for (db in unique(all_ora$database)) {
    out_png <- file.path(
      figs,
      paste0("PRJNA913947_03B_somatic_ORA_", q, "_", db, ".png")
    )
    plot_top(all_ora, q, db, out_png)
  }
}

report <- file.path(status, "PRJNA913947_stage_03B_database_somatic_pathway_ORA_status.txt")

sink(report)
cat("PRJNA913947 Stage 03B database-driven somatic pathway ORA\n")
cat("Generated:", as.character(Sys.time()), "\n\n")
cat("Input gene counts\n")
cat("background_all_annotated_genes:", length(background), "\n")
cat("functional_extended:", length(query_ext), "\n")
cat("strict_HIGH_MODERATE:", length(query_strict), "\n")
cat("functional_extended_no_caution_sensitivity:", length(query_nocaution), "\n\n")
cat("MSigDB gene set summary\n")
print(set_summary)
cat("\nORA summary by query/database\n")
print(summary_by_query_db)
cat("\nTop ORA results\n")
print(top_results %>% select(query_label, database, gs_name, overlap_n, gene_set_size, odds_ratio, pvalue, qvalue_BH_within_database, overlap_genes) %>% head(80))
cat("\nFDR <= 0.25 results\n")
print(sig_results %>% select(query_label, database, gs_name, overlap_n, gene_set_size, odds_ratio, pvalue, qvalue_BH_within_database, overlap_genes) %>% head(120))
sink()

cat("all_results", all_out, "\n")
cat("top25", top_out, "\n")
cat("FDR025", sig_out, "\n")
cat("overlap_genes_long", overlap_long_out, "\n")
cat("summary_by_query_database", summary_out, "\n")
cat("report", report, "\n")
RS

echo
echo "=== ORA SUMMARY BY QUERY/DATABASE ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_somatic_database_ORA_summary_by_query_database.tsv"

echo
echo "=== TOP ORA RESULTS ==="
column -t -s $'\t' "$OUTDIR/PRJNA913947_somatic_database_ORA_top25_per_query_database.tsv" | head -80

echo
echo "=== FDR <= 0.25 RESULTS ==="
if [[ -s "$OUTDIR/PRJNA913947_somatic_database_ORA_FDR025.tsv" ]]; then
  column -t -s $'\t' "$OUTDIR/PRJNA913947_somatic_database_ORA_FDR025.tsv" | head -120
else
  echo "No FDR <= 0.25 terms."
fi

echo
echo "=== OUTPUT FILES ==="
find "$OUTDIR" -maxdepth 1 -type f -printf "%f\t%k KB\n" | sort

echo
echo "=== FIGURES ==="
find "$FIGS" -maxdepth 1 -type f -name "PRJNA913947_03B_somatic_ORA_*png" -printf "%f\t%k KB\n" | sort

echo
echo "=== DONE: STAGE 03B ==="
date
