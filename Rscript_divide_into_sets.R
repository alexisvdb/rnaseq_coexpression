#### What does this code do? #### 

#This code reads in the sample and study annotation data, and divides the
#studies into sets of studies that have overlapping cell type annotations. Why
#is this needed? In next steps we will try to remove batch effects from the
#data. To batch effect correction methods, for each sample we need to give as
#input the batch (=study) that it belongs to, and the cell type. If there is a
#perfect confounding between batches and cell types, this will result in errors.
#To avoid those errors, we split the data into sets of studies that are
#internally not perfectly confounded. In next steps, we will remove batch
#effects from each set.



#### process user input ####
args <- commandArgs(TRUE)
sample.to.study.file <- args[1] # ex: "rawdata/annotation_data.txt"
cells.to.count.file  <- args[2] # ex: "rawdata/cell_types_vs_index.txt"
study.set.outfile    <- args[3] # ex: "rawdata/study_sets.txt"



#### Read in annotation data etc ####
# samples vs studies vs cell type
final_anno <- read.table(annotation.file, sep="\t", row.names=1, header=T, quote="\"")

# cell type vs sample count
cell2sample_count <- read.table(cells.to.count.file, sep="\t", header=T)



#### Some simple processing ####
# retain only cell types with at least 10 samples
cells <- cell2sample_count$cell_or_tissue[ cell2sample_count$sample_count >= 10]
final_anno <- subset(final_anno, is.element(final_anno[,2],cells))
studies    <- unique(final_anno$study_id)



#### Separate the studies into sets with overlapping cell types ####
studies_copy <- studies
set_index <- 1
sets <- rep(0, length(studies))
names(sets)<-as.character(studies)
while(length(studies_copy)>0){

	study <- studies_copy[1]
	in_set <- study
	set_increased_boo <- TRUE

	while(set_increased_boo){
		# get the cell types of all studies in the set
		cells_tmp <- unique(as.vector(subset(final_anno[,2], is.element(final_anno[,1],in_set))))

		# get the studies that have a sample of one of these cell types
		studies_tmp <- unique(as.vector(subset(final_anno[,1], is.element(final_anno[,2],cells_tmp))))

		if(length(setdiff(studies_tmp,in_set))>0) {
			set_increased_boo <- TRUE	
		} else {
			set_increased_boo <- FALSE
		}

		in_set <- unique(c(studies_tmp,in_set))

	}
	sets[as.character(in_set)] <- set_index
	studies_copy <- setdiff(studies_copy,in_set)
	set_index <- set_index + 1

}



#### write output ####
write.table(file = study.set.outfile, sets, col.names = FALSE, quote = FALSE, sep = "\t")
