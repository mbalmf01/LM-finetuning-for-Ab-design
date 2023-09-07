#!/bin/bash

# List of values for the --min-seq-id parameter
min_seq_id_values=("0.8" "0.9" "0.7")

# Loop through each value for --min-seq-id and extract data
for min_seq_id in "${min_seq_id_values[@]}"; do
    sudo mmseqs easy-cluster 230716_scfv_10000.fasta clusterRes new_tmp --min-seq-id "$min_seq_id" -c 0.8 --cov-mode 1
    python /content/drive/MyDrive/msc_project/antibody_analysis/mmseq_processor.py "$min_seq_id"
done