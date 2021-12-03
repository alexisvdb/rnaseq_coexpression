---
title: "Evaluation of critical data processing steps for reliable prediction of gene co-expression from large collections of RNA-seq data"
---

# Citation
This code accompanies the paper "Evaluation of critical data processing steps for reliable prediction of gene co-expression from large collections of RNA-seq data" by Alexis Vandenbon (bioRxiv doi: https://doi.org/10.1101/2021.03.11.435043). If you found this code useful in your research, please cite this paper.

This code includes R, C, and some Perl scripts. Code has been written with as few dependencies as possible, and using base R wherever possible. 

# Input data
A few of the input datasets are also included in this repository. Larger datasets for both human and mouse RNA-seq samples can be found on figshare, with DOIs doi.org/10.6084/m9.figshare.14178446.v1 and doi.org/10.6084/m9.figshare.14178425.v1. These datasets include: 

1. raw read counts of genes in all RNA-seq samples
2. the same RNA-seq data after UQ normalization and batch effect correction using ComBat, which is in general the best workflow according to our study
3. annotation data assigning a study ID and cell type or tissue to each sample
4. a list of each cell type or tissue included in the datasets along with its sample count.


# Summary of the workflow
For details, please refer to the paper mentioned above. In brief, this code allows you to 

1. To be added later
2. etc
3. etc

## Step 1: Normalization
Read in the entire (human or mouse) dataset and normalize it using a method of choice ("quantile", "rlog", "cpm", "tmm", "med", "uq", or  "none"). Output are normalized values (log10).

Example usage:
```{bash}
Rscript Rscript_normalization.R <input_count_file> <normalization_method> <normalized_output_file>
```

Example: normalize the 8,796 human RNA-seq samples using Upper Quartile ("uq") normalization, and output to dat_norm_uq_log10.txt.
```{bash}
Rscript Rscript_normalization.R raw_count_data_filtered.txt uq dat_norm_uq_log10.txt
```
