#### What does this code do? #### 

# This code reads through the results of TFBS enrichment analysis and returns
# the fraction of genes that have at least 1 significantly enriched TFBS motif.


#### process user input ####
args <- commandArgs(TRUE)
file_prefix		= args[1]
output_file		= args[2]

# file_prefix <- "results_tfbs_enrichment/tfbs_"
# output_file <- "processed_results/tfbs_uq_combat_liver.txt"

file_seed = paste(file_prefix,"*", sep="")
files <- Sys.glob(file_seed)

temp_counts <- c()

#### read in data ####
for(f in files){
  dat <- as.vector(read.table(f, sep="\t", header=F)[,2])
  
  temp_counts <- c(temp_counts,dat)
}


#### get the result we want ####
result <- as.vector(mean(temp_counts>0))


#### write output to file ####
write.table(file=output_file, result, quote=F, sep="\t", row.names = FALSE, col.names = FALSE)

