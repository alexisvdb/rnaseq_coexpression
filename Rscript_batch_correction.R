#### What does this code do? #### 

# This code reads in the gene expression data and removes batch effects according 
# to the method of choice. For removeBatchEffect and ComBat, the input is assumed 
# to be log10 values, and the output is also log10 values. For ComBat-seq, input
# should be read counts, and the output too is read counts.



#### process user input ####
args <- commandArgs(TRUE)
input.expression.file  <- args[1] # ex: "dat_norm_rlog_log10.txt"
batch.method           <- args[2] # ex: "removeBatchEffect"
sample.annotation.file <- args[3] # ex: "rawdata/annotation_data.txt"
study.set.file         <- args[4] # ex: "rawdata/study_sets.txt"
outfile                <- args[5] # ex: "dat_batch_rlog_log10.txt"

# allowed values 
batch.allowed         <- c("none", "removeBatchEffect", "combat", "combat_seq")

# check if input values are allowed values
if(!batch.method %in% batch.allowed){
  stop("'batch.method' must be one of: none, removeBatchEffect, combat, combat_seq")
}



#### read in R functions ####
source("Rfunctions_batch_correction.R")



#### read in gene expression data ####
# this is very slow
# raw <- read.table(file = raw.expression.file, sep = "\t", header = TRUE)

# this is slightly faster
con <- file(description = input.expression.file)
temp <- readLines(con, 1)         # read one line
cols <- strsplit(temp, split = "\t") # get the columns
ncols <- length(cols[[1]])        # get number of columns

# note: using colClasses "numeric" instead of "integer" because in some cases
# the data might not be integers
dat <- read.table(input.expression.file, sep = "\t", check.names = FALSE, row.names = 1, 
                  header = 1, nrow = 100000, 
                  colClasses = c("character",rep("numeric", ncols)), comment.char="")
close(con)



#### read in the sample annotations ####
annotation <- read.table(sample.annotation.file, sep = "\t", quote = '"', row.names = 1, header=1)
study.sets <- read.table(study.set.file, row.names = 1)



#### batch correction of the data ####
result <- batch_correction(dat = dat, method = batch.method, study.sets, annotation)
# note: 
# - for ComBat and removeBatchEffect, the input and output should be log values
# - for ComBat-seq, the input and output should be integers (not log values)


#### write output ####
# round to 6 digits if the method is ComBat or removeBatchEffect
# not needed for ComBat-seq (should be integers, no rounding needed)
if(batch.method=="removeBatchEffect" | batch.method == "combat")
  result <- round(result, digits=6)

# write to file
write.table(file=outfile, result, sep="\t", quote=F)

