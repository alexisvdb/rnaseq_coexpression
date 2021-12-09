#### What does this code do? #### 

# This code reads through the results of GO term enrichment analysis and returns
# the fraction of genes that have at least 1 significantly enriched GO term.


#### process user input ####
args <- commandArgs(TRUE)
file_prefix		= args[1]
output_file		= args[2]

file_seed = paste(file_prefix,"*", sep="")
files <- Sys.glob(file_seed)
	
temp_counts <- matrix(nrow=0, ncol=3)
colnames(temp_counts) <- c("MF", 	"BP", "CC")


#### read in data ####
for(f in files){
	dat <- read.table(f, sep="\t", header=T)
	
	temp_counts <- rbind(temp_counts, dat[,c(1,3,5)])
}


#### get the result we want ####
# the fraction of genes with an enriched term
result <- matrix(apply(temp_counts>0,2,mean), nrow=1)
colnames(result) <- c("MF", 	"BP", "CC")


#### write output to file ####
write.table(file=output_file, result, quote=F, sep="\t", row.names = FALSE)
