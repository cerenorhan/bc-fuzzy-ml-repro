#!/usr/bin/env bash

# Repository / project root
export PROJECT_ROOT="${PROJECT_ROOT:-/path/to/project_root}"

# Local institutional sequencing data
export LOCAL_DATA_ROOT="${LOCAL_DATA_ROOT:-/path/to/local_data}"

# Public sequencing datasets
export PUBLIC_DATA_ROOT="${PUBLIC_DATA_ROOT:-/path/to/public_data}"

# Large temporary/intermediate analysis storage
export SCRATCH_ROOT="${SCRATCH_ROOT:-/path/to/scratch}"

# Reference genomes and annotation resources
export REFERENCE_ROOT="${REFERENCE_ROOT:-/path/to/references}"

# Manuscript-oriented derived results
export PAPER_RESULTS_ROOT="${PAPER_RESULTS_ROOT:-${PROJECT_ROOT}/paper_results}"

# Optional second scratch/storage location
export SECONDARY_SCRATCH_ROOT="${SECONDARY_SCRATCH_ROOT:-/path/to/secondary_scratch}"



# Optional execution notifications.
# Set to 1 only if a compatible notification helper is configured.
export ENABLE_NOTIFICATIONS="${ENABLE_NOTIFICATIONS:-0}"
export NOTIFICATION_HELPER="${NOTIFICATION_HELPER:-}"
