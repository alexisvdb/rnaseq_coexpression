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
5. calculation of gene-gene coexpression
6. GO term and TFBS enrichment analysis


## Step 1: Normalization
Read in the entire (human or mouse) dataset and normalize it using a method of choice ("quantile", "rlog", "cpm", "tmm", "med", "uq", or  "none"). Output are normalized values (log10).

Example usage:
```{bash}
Rscript Rscript_normalization.R <input_count_file> <normalization_method> <normalized_output_file>
```

Example: normalize the 8,796 human RNA-seq samples using Upper Quartile ("uq") normalization, and output to `dat_norm_uq_log10.txt`.
```{bash}
Rscript Rscript_normalization.R raw_count_data_filtered.txt uq dat_norm_uq_log10.txt
```

Please note that the output will be several GB in size.


## Step 2: Split the samples into sets
In step 3 we will attempt to remove batch effects from the data. The batch effect correction methods will attempt to remove technical sources of variation while retaining biological sources of variation. In this case, technical sources of variation are batch effects between studies, and biological sources of variation are different cell types or tissues which the samples originate from. To successfully remove batch effects, we will have to give as input to the batch effect correction methods the data showing for each sample 1) what study it came from (=the batch) and 2) what cell type or tissue it came from (=the biological signal).

However, in practice many studies focus on one single cell type or tissue. Some include samples for a few cell types or tissues, and only a minority contains data for many cell types. As a result, we end up with sets of studies (batches) between which there is no overlap in cell type or tissues. Such confounding between technical and biological sources of variation results in problems for the batch correction methods.

As a solution to this problem, we can separate the data into sets of studies that do have overlapping cell types and tissue samples. This is what we do in this step.

Example usage:
```{bash}
Rscript Rscript_divide_into_sets.R <sample_annotation> <cell_types> <study_set_file>
```

The output is in `<study_set_file>`, in which each sample will have been assigned to a set.

Under `example_input/` there is example input for our human dataset.

```{bash}
Rscript Rscript_divide_into_sets.R example_input/annotation_data.txt example_input/cell_types_vs_index.txt study_sets.txt
```


## Step 3: Batch effect correction
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

## Step 4: Extract the data for a particular cell type or tissue

After normalization and batch effect correction, let's extract the data for a particular cell type or tissue. That data will later on be used for estimating gene co-expression in that tissue/cell type.

The script `Rscript_get_data_for_cell_type.R` not only extracts that data we want, but also filters out genes with very low expression in our target cell type, and also outputs several other files that will be used in the downstream analysis. It has several input parameters and output files:

- `raw.read.count.file` = the file with the raw (not normalized) read counts per gene per samples
- `processed.expression.file` = the file with the normalized and/or batch effect corrected data
- `sample.annotation.file` = annotation data for each samples
- `target.cell.type` = the cell type or tissue of interest
- `biomart.file` = a mapping between Ensembl gene ids and Entrez (NCBI) gene ids
- `process.to.ranks` = "TRUE" or "FALSE". See explanation below
- `output.gene.file` = output file for a list of Ensembl gene ids that are present in the final data
- `output.entrez.id.file` = output file for a list of Entrez (NCBI) gene ids that are present in the final data
- `output.gene.to.entrez.id.file` = output file with a mapping between Ensembl and Entrez ids
- `output.expression.data.file` = output file for the gene expression data for the cell type or tissue of interest

About the `process.to.ranks` input parameter: In Step 5 gene-gene correlation of expression will be calculated using a C script. This script is fast, but can  calculate only Pearson's correlation, not Spearman's correlation. If we want it to calculate Spearman's correlation coefficient, we can convert the gene expression data first to ranks, by setting `process.to.ranks` to "TRUE". Giving this rank data as input to the C script is equivalent to calculating Spearman's correlation.


Example usage:
```{bash}
Rscript Rscript_get_data_for_cell_type.R <raw.read.count.file> <processed.expression.file> <sample.annotation.file> <target.cell.type> <biomart.file> <process.to.ranks> <output.gene.file> <output_entrez_id_file> <output.gene.to.entrez.id.file> <output.expression.data.file>
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


## Step 5: Calculation of gene-gene coexpression
### Step 5.1: Calculating correlation of expression between pairs of Ensembl genes
Directory `src_massCorrelation/` contains several C scripts for calculating correlation of expression between pairs of gene on a genome-wide scale. 

To compile the C code, prepare a directory `bin_massCorrelation/` and run `make` inside the `src_massCorrelation directory`. Finally, we need to copy the shell script `process_to_correlation.sh` to the `bin_massCorrelation/` directory.

```{bash}
mkdir bin_massCorrelation
cd src_massCorrelation/
make
cp process_to_correlation.sh ../bin_massCorrelation/
```

To calculate the gene-gene coexpression values, we need to run the `process_to_correlation.sh` script.

Example usage:
```{bash}
./bin_massCorrelation/process_to_correlation.sh <expression_data> <binary_correlation_file>
```
The `<expression_data>` is the cell type or tissue-specific data which we prepared in Step 4. The output is correlation values in a binary format, in `<binary_correlation_file>`. For our data, each binary output file is about 1 GB in size.

Example usage on the human liver data:
```{bash}
./bin_massCorrelation/process_to_correlation.sh data_liver.txt liver_correlation_pearson.bin
```

Similarly, we can get the Spearman correlation by using the ranked data as input:
```{bash}
./bin_massCorrelation/process_to_correlation.sh data_liver_RANKED.txt liver_correlation_spearman.bin
```

### Step 5.2: Processing to correlation of expression between pairs of Entrez genes
The above steps give us the correlation between pairs of Ensembl genes. For the downstream analysis (evaluation of quality of coexpression estimates) we want to use not Ensembl but Entrez (NCBI) ids. Therefore, we convert the Ensembl-based correlations to Entrez-based correlations:

Example usage:
```{bash}
./bin/probe2symbolScore <ensembl_gene_id_file> <entrez_id_file> <gene_to_entrez_id_file> < <binary_correlation_file> > <binary_correlation_entrez_file>
```

For our human liver correlation data:
```{bash}
./bin/probe2symbolScore ensembl_gene_list.txt ncbi_gene_list.txt ensembl_to_ncbi_table.txt < liver_correlation_pearson.bin >liver_correlation_pearson_entrez.bin
```

If the above commands result in error messages, they might be caused by differences in new line and carriage return characters between different operating systems. These errors might be fixed by running [`dos2unix`](https://en.wikipedia.org/wiki/Unix2dos) on the input files:

For example:
```{bash}
dos2unix ensembl_gene_list.txt
dos2unix ncbi_gene_list.txt
dos2unix ensembl_to_ncbi_table.txt
```

### Step 5.3: Splitting correlation data into one file per gene

The binary correlation files that we got so far contain the correlation between all pairs of genes in 1 single file. But for the downstream analysis, it is more useful to have 1 file per single gene. Here we split the binary correlation data into smaller binary files, one per gene.

Warning: this obviously results in a LARGE amount of files!!

Example usage:
```{bash}
./bin/makeBinTable4 <entrez_id_file> <dir_for_split_binary_files> <binary_correlation_entrez_file>
```

`<dir_for_split_binary_files>` is a temporary directory made just for storing the many small files.

For our human liver example:
```{bash}
mkdir binary_files_split_per_gene # make a temp directory
./bin/makeBinTable4 ncbi_gene_list.txt binary_files_split_per_gene liver_correlation_pearson_entrez.bin
```


## Step 6: GO term and TFBS enrichment analysis

At this point we have a directory with gene co-expression data for all Entrez genes in our dataset, split into 1 binary file per gene. To estimate the quality of this correlation data, we will extract for every gene the set of 100 most highly correlated genes, and check if those 100 genes are enriched for Gene Ontology (GO) terms, and if their promoters are enriched for predicted transcription factor binding sites (TFBSs).

The basic concept is that in high-quality gene coexpression data, gene with common functional annotations (=GO) terms should in general tend to be more frequently correlated than in low-quality gene coexpression data. Similarly, in high-quality gene coexpression data, coexpressed genes should have shared regulatory motifs (=TFBSs) in their promoter regions more frequently than in low-quality coexpression data.

### Step 6.1: Preparation
Basically, we want to look at the data of each gene at the time. In practice, we will process genes in sets of 1000. 

Split genes into sets of 1000:
```{bash}
mkdir temp_split_ncbi # a temporary directory to store the sets
split -l 1000 -d ncbi_gene_list.txt temp_split_ncbi/ncbi_
```
This should result in about 20 files like `temp_split_ncbi/ncbi_00`, `temp_split_ncbi/ncbi_01`, etc, each containing 1000 Entrez ids.


### Step 6.2: Preparing GO data

Various input datasets were obtained:

- Basic version of the Gene Ontology: go-basic.obo from [Gene Ontology website](http://geneontology.org/)
- Human gene annotation data: goa_human.gaf from [the EBI database] (ftp://ftp.ebi.ac.uk/pub/databases/GO/goa/HUMAN/goa_human.gaf.gz)
- Mouse gene annotation data: gene_association.mgi from [the Mouse Genome Informatics website](http://www.informatics.jax.org/)

Other processed input data was obtained from Vandenbon et al., PNAS, 2016: a mapping of GO terms to their parent terms (`go_mapped_to_tree_MF.txt`, `go_mapped_to_tree_BP.txt`, and `go_mapped_to_tree_CC.txt`).

In Step 4 we extracted data for our cell type and tissue of interest, which also involved filtering out genes with very low expression values in that cell type or tissue. As a result, our data does not include all genes. For the GO term enrichment analysis too, we want to focus only on genes that are present in our data. Some filtering is necessary.

We filter out GO data for genes not included in the processed human liver data, for Biological Process (BP), Molecular Function (MF) and Cellular Component (CC):
```{bash}
perl GO/process_go_annotations_to_entrez_ids_present_in_data.pl GO/go_mapped_to_tree_MF.txt GO/go-basic.obo GO/goa_human.gaf GO/symbol2entrez_pairs_human.txt ncbi_gene_list.txt >ncbi2go_MF.txt
perl GO/process_go_annotations_to_entrez_ids_present_in_data.pl GO/go_mapped_to_tree_BP.txt GO/go-basic.obo GO/goa_human.gaf GO/symbol2entrez_pairs_human.txt ncbi_gene_list.txt >ncbi2go_BP.txt
perl GO/process_go_annotations_to_entrez_ids_present_in_data.pl GO/go_mapped_to_tree_CC.txt GO/go-basic.obo GO/goa_human.gaf GO/symbol2entrez_pairs_human.txt ncbi_gene_list.txt >ncbi2go_CC.txt
```
Note that the resulting ncbi2go_MF.txt (same for BP and CC) will be different for each cell type or tissue, because the filtered-out genes will be slightly different. These files contain 3 columns: a column of Entrez (NCBI) gene ids, all GO terms associated with that gene (including parental nodes), and the domain of the Gene Ontology of each term (F for Molecular Function, P for Molecular Process, and C for Cellular Component). Each gene-to-term annoation is 1 line.

For each GO term, we count how many genes in our dataset are associated with it:
```{bash}
sort -u ncbi2go_MF.txt | cut -f 2 | sort | uniq -c | awk '{print $2"\t"$1}' >go_counts_MF.txt
sort -u ncbi2go_BP.txt | cut -f 2 | sort | uniq -c | awk '{print $2"\t"$1}' >go_counts_BP.txt
sort -u ncbi2go_CC.txt | cut -f 2 | sort | uniq -c | awk '{print $2"\t"$1}' >go_counts_CC.txt
```
The resulting files contain 1 line per GO term, including the GO id of the term and the count of genes associated with it.

Finally, we get the total number of GO terms in the above 3 GO sets, which we will use for correction for multiple testing later on:
```{bash}
cat go_counts_* | wc -l >go_counts_total.txt
```

To give an idea, for the human liver data processed using Upper Quartile normalization and ComBat batch effect correction, we end up with about 21k GO terms.
	

### Step 6.3: Running GO term enrichment analysis

We run the actual GO term enrichment analysis using `Rscript_GO_enrichment.R`.

Example usage:
```{bash}
Rscript Rscript_GO_enrichment.R <ncbi.subset.file> <bin.data.dir> <outfile> <go.data.dir> <total.go.count_file> <ncbi.id.file>
```
The `<ncbi.subset.file>` refers to the sets of 1000 gene ids which we prepared in Step 6.1. `<bin.data.dir>` refers to the directory that includes the many small binary correlation files, one for each Entrez gene. `<go.data.dir>` is the directory where the GO-related data is located (See step 6.2).

For the first set of 1000 Entrez ids (in file temp_split_ncbi/ncbi_00), the command would be:
```{bash}
mkdir tmp_GO_output # make a temporary dir to put all GO related results
Rscript Rscript_GO_enrichment.R temp_split_ncbi/ncbi_00 binary_files_split_per_gene tmp_GO_output/go_00 GO/ GO/total.go.count_file ncbi_gene_list.txt
```
This should be done for each of the (about 20) sets of 1000 Entrez IDs.




