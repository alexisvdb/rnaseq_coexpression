#### What does this code do? #### 

# This code takes a subset of genes as input, and for each gene:
# - gets the top 100 highly correlated genes
# - uses those 100 genes to perform GO term enrichment analysis
# - saves the result to file



#### process user input ####
args <- commandArgs(TRUE)

ncbi.subset.file		<- args[1] # file with a subset of ncbi ids
bin.data.dir      	<- args[2] # dir with the binary correlation data of each gene
outfile 		        <- args[3] # for output
go.data.dir 	      <- args[4] # dir where ncbi2go_* and go_count_* files are located
total.go.count_file <- args[5] # file with the total no. of GO terms
ncbi.id.file	      <- args[6] # file with all ncbi gene ids



#### read input data ################################

# read NCBI ID list
ncbi.ids <- as.vector(read.table(ncbi.id.file)[,1])

# read in GO data and process
read_gene2go = function(type){
	gene2go.file <- paste(go.data.dir,"/ncbi2go_",type,".txt", sep="")
	# read gene2go associations
	gene2go <- read.table(gene2go.file)
	gene2go <- unique(gene2go)
	gene2go
}
read_go_count = function(type){
	go.count.file <- paste(go.data.dir,"/go_counts_",type,".txt", sep="")
	# get genome-wide counts of each GO
	genome <- read.table(go.count.file, row.names=1)
	genome
}

total.go.ids <- as.numeric(
  system( paste0("cat ",total.go.count_file), intern=TRUE )
)

# read in gene to GO annotations
gene2go.MF <- read_gene2go("MF")
gene2go.BP <- read_gene2go("BP")
gene2go.CC <- read_gene2go("CC")

# get the total number of genes with at least 1 GO annotation
size.genome.MF <- length(unique(gene2go.MF[,1]))
size.genome.BP <- length(unique(gene2go.BP[,1]))
size.genome.CC <- length(unique(gene2go.CC[,1]))

# read in GO term counts
genome.MF <- read_go_count("MF")
genome.BP <- read_go_count("BP")
genome.CC <- read_go_count("CC")


#### function for performing the enrichment analysis ####
go_enrichment = function(genes=genes, size.genes=size.genes,inputDir, type){
  
	# specify the correct background information for this type
	gene2go <- switch(
		type,
		MF = gene2go.MF,
		BP = gene2go.BP,
		CC = gene2go.CC
	)
	size.genome <- switch(
		type,
		MF = size.genome.MF,
		BP = size.genome.BP,
		CC = size.genome.CC
	)
	genome <- switch(
		type,
		MF = genome.MF,
		BP = genome.BP,
		CC = genome.CC
	)

	# get GOs for each gene, and add them to a vector
	all.gos <- c()
	for(i in 1:size.genes){
		temp <- unique(as.vector(gene2go[gene2go[,1]==genes[i],2]))
		all.gos <- c(all.gos,temp)
	}# end foreach gene
	tab <- table(all.gos);
	
	
	# get subset of genome-wide counts corresponding to the GOs in "tab"
	tab.ref <- genome[names(tab),]

	# calculate probabilities of enrichment
	pvals <- phyper(tab - 1, tab.ref, size.genome - tab.ref, size.genes, lower.tail=F)
	# calculated the expected counts of eacg GO, given the size of the input set and the genomic counts
	expected.counts <- tab.ref * size.genes / size.genome
	# the fold enrichment of counts in the input set compared to expected
	fold.enrichment <- tab / expected.counts

	# make a result table to print
	result <- matrix(0, nrow=length(tab), ncol=5)
	row.names(result) <- names(tab)
	colnames(result) <- c("observed count", "expected count", "fold enrichment", "p value", "corrected p value")
	result[,1] <- tab
	result[,2] <- round(expected.counts,digits=3)
	result[,3] <- signif(fold.enrichment,digits=3)
	result[,4] <- signif(pvals,digits=3)
	result[,5] <- signif(pvals * total.go.ids) # this one is corrected for multiple testing

	# if the corrected p value is higher than 1: set it to 1
	result[result[,5]>1, 5] <- 1

	# sort this table
	o <- order(result[,4])
	result <- result[o,]

	return(as.matrix(result))
}



#### run through set of IDs, and do enrichment analysis ####
id.list <- read.table(ncbi.subset.file)[,1]

output <- matrix("-", nrow=length(id.list), ncol=6)
row.names(output) <- id.list
colnames(output) <- c("MF_count","MF_terms","BP_count","BP_terms","CC_count","CC_terms")


#### the main analysis #####
# for each gene in the id.list:
# - read in its binary correlation data
# - get the top 100 (not duplicated; excluding the gene itself) correlated genes
# - perform GO term enrichment analysis using those genes
# - process results

# overstated number of items to read
parts.to.read <- 100000

for(i in 1:length(id.list)){
	
	####################################
	### read in binary file, get top 100 correlated genes
	####################################
	
	id <- id.list[i]
	bin.file <- paste(bin.data.dir,"/",id, ".bin", sep="")
	
	if(!file.exists(bin.file))
		next
	
	# open the binary file
	to.read = file(bin.file, open="rb")
	# rb is reading in binary mode
	
	correlation <- readBin(to.read, double(), size=4, n=parts.to.read)
	# double() is the type of data
	# size is 4: 4 bytes for the correlation
	# n is the number of "lines" to read. This can be an overestimate. If you use the default (1) only 1 entry is read.
	
	close(to.read)
	
	o <- order(correlation, decreasing=T)
	not.duplicated <- !duplicated(correlation[o[1:1000]])

	top.correlated.ncbi <- ncbi.ids[o[not.duplicated][2:101]]
	
	genes <- top.correlated.ncbi

	# get the number of genes in this set; should be 100
	size.genes <- length(genes)


	#########################
	### run the analysis for each of the GO spaces
	#########################

	# for molecular function
	res.MF <- go_enrichment(genes, size.genes, inputDir, "MF")
	# for biological process
	res.BP <- go_enrichment(genes, size.genes, inputDir, "BP")
	# for cellular component
	res.CC <- go_enrichment(genes, size.genes, inputDir, "CC")
	
	#########################
	### process results 
	#########################
	
	# MF
	enriched.terms <- subset(row.names(res.MF), res.MF[,"corrected p value"] < 0.01)
	output[i,1] <- as.character(length(enriched.terms))
	output[i,2] <- paste(enriched.terms,collapse=";")
	
	# BP
	enriched.terms <- subset(row.names(res.BP), res.BP[,"corrected p value"] < 0.01)
	output[i,3] <- as.character(length(enriched.terms))
	output[i,4] <- paste(enriched.terms,collapse=";")
	
	# CC
	enriched.terms <- subset(row.names(res.CC), res.CC[,"corrected p value"] < 0.01)
	output[i,5] <- as.character(length(enriched.terms))
	output[i,6] <- paste(enriched.terms,collapse=";")
	
	
}# end foreach ID


#### print output ####
write.table(file = outfile, output, sep = "\t", quote = FALSE)
