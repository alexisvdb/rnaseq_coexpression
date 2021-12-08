#### What does this code do? #### 

# This code extracts from the larger processed gene expression data the data for
# a particular cell type or tissue of interest. If needed it converts the gene
# expression values of each gene into ranks, which can be used to calculate the
# Spearman correlation in the downstream analysis. The script also prints out
# lists of gene ids that are present in the data.


#### process user input ####
args <- commandArgs(TRUE)
raw.read.count.file 		       = args[1]
processed.expression.file 		 = args[2]
sample.annotation.file 	       = args[3]
target.cell.type 	             = args[4]
biomart.file			             = args[5]
process.to.ranks	             = args[6]
output.gene.file               = args[7]
output_entrez_id_file 	       = args[8]
output.gene.to.ncbi.id.file    = args[9]
output.expression.data.file		 = args[10]



#### check user input ####
if(process.to.ranks!="TRUE" & process.to.ranks!="FALSE"){
  stop("'process.to.ranks' must be either TRUE or FALSE")
}
if(!file.exists(raw.read.count.file)){
  stop("Can't find 'raw.read.count.file' (\"",raw.read.count.file,"\"). Check your input.")
}
if(!file.exists(processed.expression.file)){
  stop("Can't find 'processed.expression.file' (\"",processed.expression.file,"\"). Check your input.")
}
if(!file.exists(sample.annotation.file)){
  stop("Can't find 'sample.annotation.file' (\"",sample.annotation.file,"\"). Check your input.")
}

process.to.ranks = as.logical(process.to.ranks)

#### read in data ####

# read in annotation data
annotation <- read.table(sample.annotation.file, sep = "\t", quote = '"', row.names = 1, header=1)

# check if the target cell type is actually present in the annotation data
target.samples <- subset(rownames(annotation), annotation$cell_or_tissue == target.cell.type)
if(length(target.samples) == 0){
  stop("No samples present for target cell type ",target.cell.type," in the annotation data. Check your input.")
}

# read in biomart output file
biomart <- read.table(biomart.file,sep="\t", header=T)


# read in raw count data
# this will be used for some filtering lateron
raw <- read.table(file=raw.read.count.file, sep="\t", header=T)

# read in processed expression data
con <- file(description=processed.expression.file)
temp <- readLines(con, 1) # Read one line
cols <- strsplit(temp,split="\t")
ncols <- length(cols[[1]])
processed.data <- read.table(processed.expression.file, sep="\t", check.names=F, row.names=1, header=1, nrow=100000, colClasses=c("character",rep("numeric",ncols)), comment.char="")
close(con)



#### process data for target cell type ####
target.col.names <- intersect(target.samples, colnames(processed.data))
if(length(target.col.names) == 0){
  stop("No target columns could be found in the processed data. Check your input.")
}

# get data of interest
target.processed.data <- processed.data[,target.col.names]
target.raw.data <- raw[,target.col.names]
sample.count <- ncol(target.processed.data)

# get genes present in data
genes <- rownames(processed.data)

# remove unnecessary data
rm(processed.data)
rm(raw)
gc()

# filter out genes with low raw counts or low processed expression values
# point 1: get the max read count for each gene
max.tag.count <- apply(target.raw.data,1,max)

# point 2: in how many samples does each gene have really low read counts?
sample.count.low.tag.count <- apply(target.raw.data < 10,1, sum)

# filter out genes with too low tag counts
genes.filtered.out <- genes[
  sample.count.low.tag.count > 0.8*sample.count & max.tag.count < 50
  | sample.count.low.tag.count >= 0.9*sample.count
]
genes.retained <- setdiff(genes, genes.filtered.out)

# get data for the remaining genes
target.processed.data <- target.processed.data[genes.retained,]

# convert data to ranks if output should be ranked (for Spearman correlation)
# unfortunately, this takes time. If you know a more elegant way, please let me 
# know (alexis.vandenbon@gmail.com)
if(process.to.ranks){
  message("# converting expression data to ranks...")
  for(i in 1:nrow(target.processed.data)){
    if(i%%5000==0){
      message("# processing gene ",i," of ",nrow(target.processed.data))
    }
    target.processed.data[i,] <- rank(target.processed.data[i,])
  }
}


#### process some gene symbol data ####

# ensembl vs ncbi gene ids
biomart <- subset(biomart, !is.na(biomart[,2]))
ensembl.ids <- genes.retained

ncbi.ids <- c()
# I am sure that there are more elegant ways to do this, but...
for(i in 1:length(ensembl.ids)){
  tmp <- biomart[biomart[,1]==ensembl.ids[i],2]
  ncbi.ids[i] <- tmp[1]
}
ensembl2entrez <- cbind(ensembl.ids,ncbi.ids)
ensembl2entrez <- subset(ensembl2entrez, !is.na(ensembl2entrez[,2]))

ncbi.ids <- sort(as.numeric(unique(ensembl2entrez[,2])))



#### save output ####
write.table(file = output.expression.data.file, target.processed.data, sep = "\t", quote = FALSE)
write.table(file = output.gene.file, genes.retained, quote=FALSE, row.names=FALSE, col.names=FALSE)
write.table(file = output.gene.to.ncbi.id.file, ensembl2entrez, quote=FALSE, row.names=FALSE, col.names=FALSE, sep="\t")
write.table(file = output.ncbi.id.file, ncbi.ids, quote=FALSE, row.names=FALSE, col.names=FALSE, sep="\t")


