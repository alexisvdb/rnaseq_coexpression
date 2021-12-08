# Evaluation of critical data processing steps for reliable prediction of gene co-expression from large collections of RNA-seq data"

## Citation
This code accompanies the paper "Evaluation of critical data processing steps for reliable prediction of gene co-expression from large collections of RNA-seq data" by Alexis Vandenbon (bioRxiv doi: https://doi.org/10.1101/2021.03.11.435043). If you found this code useful in your research, please cite this paper.

This code includes R, C, and some Perl scripts. Code has been written with as few dependencies as possible, and using base R wherever possible. 

## Input data
A few of the smaller input files are included in this repository as an example (see dir example_input). Larger datasets for both human and mouse RNA-seq samples can be found on figshare, with DOIs [doi.org/10.6084/m9.figshare.14178446.v1](https://figshare.com/articles/dataset/Human_data/14178446/1) and [doi.org/10.6084/m9.figshare.14178425.v1](https://figshare.com/articles/dataset/Mouse_data/14178425/1). These datasets include: 

1. raw read counts of genes in all RNA-seq samples
2. the same RNA-seq data after UQ normalization and batch effect correction using ComBat, which is in general the best workflow according to our study
3. annotation data assigning a study ID and cell type or tissue to each sample
4. a list of each cell type or tissue included in the datasets along with its sample count.


## Summary of the workflow
For details, please refer to the paper mentioned above. In brief, this code allows you to 

1. normalize a large RNA-seq gene expression dataset
2. split the samples into sets
3. remove batch effects
4. extract the data for a particular cell type or tissue

### Step 1: Normalization
Read in the entire (human or mouse) dataset and normalize it using a method of choice ("quantile", "rlog", "cpm", "tmm", "med", "uq", or  "none"). Output are normalized values (log10).

Example usage:
```{bash}
Rscript Rscript_normalization.R <input_count_file> <normalization_method> <normalized_output_file>
```

Example: normalize the 8,796 human RNA-seq samples using Upper Quartile ("uq") normalization, and output to dat_norm_uq_log10.txt.
```{bash}
Rscript Rscript_normalization.R raw_count_data_filtered.txt uq dat_norm_uq_log10.txt
```

Please note that the output will be several GB in size.


### Step 2: Split the samples into sets
In step 3 we will attempt to remove batch effects from the data. The batch effect correction methods will attempt to remove technical sources of variation while retaining biological sources of variation. In this case, technical sources of variation are batch effects between studies, and biological sources of variation are different cell types or tissues which the samples originate from. To successfully remove batch effects, we will have to give as input to the batch effect correction methods the data showing for each sample 1) what study it came from (=the batch) and 2) what cell type or tissue it came from (=the biological signal).

However, in practice many studies focus on one single cell type or tissue. Some include samples for a few cell types or tissues, and only a minority contains data for many cell types. As a result, we end up with sets of studies (batches) between which there is no overlap in cell type or tissues. Such confounding between technical and biological sources of variation results in problems for the batch correction methods.

As a solution to this problem, we can separate the data into sets of studies that do have overlapping cell types and tissue samples. This is what we do in this step.

Example usage:
```{bash}
Rscript Rscript_divide_into_sets.R <sample_annotation> <cell_types> <study_set_file>
```

The output is in <study_set_file>, in which each sample will have been assigned to a set.

Under example_input/ there is example input for our human dataset.

```{bash}
Rscript Rscript_divide_into_sets.R example_input/annotation_data.txt example_input/cell_types_vs_index.txt study_sets.txt
```


### Step 3: Batch effect correction
Read in the entire (human or mouse) normalized dataset and remove batch effects using a method of choice. Methods include limma's removeBatchEffect function, sva's Combat, and Combat-seq. This is done for each set separately (see Step 2).

Note that Combat-seq expects as input not normalized data, but read counts. So, if you want to use Combat-seq for batch effect correction, we recommend first running Steps 2 and 3, without Step 1 (Combat-seq is supposed to take care of both normalization and batch effect correction). If you still want to run a round of normalization on the Combat-seq processed data, you can run Step 1 on the output of Combat-seq.

Example usage:
```{bash}
Rscript Rcommands_batch_effect_correction.R <input_data_file> <batch_correction_method> <sample_annotation> <study_set_file> <batch_corrected_output_file>
```

For example, to treat batch effects using ComBat on the output of Upper Quartile normalization (as done in Step 1):
```{bash}
Rscript Rcommands_batch_effect_correction.R dat_norm_uq_log10.txt combat example_input/annotation_data.txt study_sets.txt dat_uq_combat_rlog_log10.txt
```

Please note that the output will be several GB in size.

Steps 1 and 3 can be used to obtain the processed data for several combinations of normalization and batch effect approaches.

### Step 4: Extract the data for a particular cell type or tissue

After normalization and batch effect correction, let's extract the data for a particular cell type or tissue. That data will later on be used for estimating gene co-expression in that tissue/cell type.

The script Rscript_get_data_for_cell_type.R not only extracts that data we want, but also outputs several other files that will be used in the downstream analysis. It has several input parameters and output files:

- raw.read.count.file = the file with the raw (not normalized) read coutns per gene per samples
- processed.expression.file = the file with the normalized and/or batch effect corrected data
- sample.annotation.file = annotation data for each samples
- target.cell.type = the cell type or tissue of interest
- biomart.file = a mapping between Ensembl gene ids and Entrez (NCBI) gene ids
- process.to.ranks = "TRUE" or "FALSE". See explanation below
- output.gene.file = output file for a list of Ensembl gene ids that are present in the final data
- output_entrez_id_file = output file for a list of Entrez (NCBI) gene ids that are present in the final data
- output.gene.to.ncbi.id.file = output file with a mapping between Ensembl and Entrez ids
- output.expression.data.file = output file for the gene expression data for the cell type or tissue of interest

About the "process.to.ranks" input parameter: In Step 5 gene-gene correlation of expression will be calculated using a C script. This script is fast, but can  calculate only Pearson's correlation, not Spearman's correlation. If we want it to calculate Spearman's correlation coefficient, we can convert the gene expression data first to ranks, by setting "process.to.ranks" to "TRUE". Giving this rank data as input to the C script is equivalent to calculating Spearman's correlation.


Example usage:
```{bash}
Rscript Rscript_get_data_for_cell_type.R <raw.read.count.file> <processed.expression.file> <sample.annotation.file> <target.cell.type> <biomart.file> <process.to.ranks> <output.gene.file> <output_entrez_id_file> <output.gene.to.ncbi.id.file> <output.expression.data.file>
```
I am sorry for the messy way of giving input parameters to these scripts. Hopefully, some day I will clean it up.

If we want to use the human data after Upper Quartile normalization followed by batch effect correction using Combat, and extract the data for liver tissue, we could do:
```{bash}
Rscript Rscript_get_data_for_cell_type.R raw_count_data_filtered.txt dat_uq_combat_rlog_log10.txt example_input/annotation_data.txt liver example_input/mart_export_human.txt FALSE ensembl_gene_list.txt ncbi_gene_list.txt ensembl_to_ncbi_table.txt data_liver.txt
```

Or, alternatively, if we are interested in Spearman's correlation, we should set FALSE to TRUE, and get the ranked data:
```{bash}
Rscript Rscript_get_data_for_cell_type.R raw_count_data_filtered.txt dat_uq_combat_rlog_log10.txt example_input/annotation_data.txt liver example_input/mart_export_human.txt TRUE ensembl_gene_list.txt ncbi_gene_list.txt ensembl_to_ncbi_table.txt data_liver_RANKED.txt
```


