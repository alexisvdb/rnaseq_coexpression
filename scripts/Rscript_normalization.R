#### What does this code do? #### 

#This code reads in the raw gene expression data and normalizes it according to
#the normalization method of choice. It also adds a small pseudo-count and takes
#log10 values. So, output data are log10 values.



#### process user input ####
args <- commandArgs(TRUE)
input.count.file     <- args[1] # ex: "rawdata/raw_count_data_filtered.txt"
normalization.method <- args[2] # ex: "rlog"
outfile              <- args[3] # ex: "dat_norm_rlog_log10.txt"

# allowed values 
normalization.allowed <- c("quantile", "rlog", "cpm", "tmm", "med", "uq", "none")

# check if input values are allowed values
if(!normalization.method %in% normalization.allowed){
  stop("'normalization.method' must be one of: quantile, rlog, cpm, tmm, med, uq, none")
}



#### read in R functions ####
source("Rfunctions_normalization.R")



#### read in data ####
# this is very slow
# raw <- read.table(file = raw.expression.file, sep = "\t", header = TRUE)

# this is slightly faster
con <- file(description = input.count.file)
temp <- readLines(con, 1)         # read one line
cols <- strsplit(temp, split = "\t") # get the columns
ncols <- length(cols[[1]])        # get number of columns

# note: using colClasses "numeric" instead of "integer" because in some cases
# the data might not be integers
raw <- read.table(input.count.file, sep = "\t", check.names = FALSE, row.names = 1, 
                  header = 1, nrow = 100000, 
                  colClasses = c("character",rep("numeric", ncols)), comment.char="")
close(con)



#### normalized the data ####
# normalize, add small pseudocount and take log10
dat.norm.log <- normalize(dat = raw, method = normalization.method)



#### write output ####
# round to 6 digits (even 6 digits is probably more than necessary)
dat.norm.log <- round(dat.norm.log, digits=6)
# write to file
write.table(file=outfile, dat.norm.log, sep="\t", quote=F)

