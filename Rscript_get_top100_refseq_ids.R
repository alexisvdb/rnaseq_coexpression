#### What does this code do? #### 

# This code returns the 100 unique Refseq ids with the highest correlation to an
# input Entrez id. This script is used inside the
# top100_refseq_motif_enrichment.sh script.



#### process user input ####
args <- commandArgs(TRUE)

single.entrez.id	  <- args[1] # a single Entrez ID (ex: "1" or "145553" etc)
bin.data.dir		    <- args[2] # directory where the binary correlations files split per gene are located
entrez.id.file		  <- args[3] # the list of Entrez IDs in the binary correlation data
entrez2refseq.file	<- args[4] # the mapping of each Entrez to 1 single refseq ID
outfile.dir			    <- args[5] # a directory to put the output file


#### read input data ####

entrez.ids <- read.table(entrez.id.file)[,1]
entrez2unique.refseq <- read.table(entrez2refseq.file, row.names=1)
entrez.in.unique.mapping.boo <- is.element(entrez.ids, row.names(entrez2unique.refseq))
entrez.in.unique.mapping <- entrez.ids[entrez.in.unique.mapping.boo]


##### run through set of IDs, get top 100 correlated Refseqs #####

# note: in this implementation, the "set" of IDs is just 1 single ID
id.list <- single.entrez.id

# overstated number of items to read from the binary file with correlation values
parts.to.read <- 100000

for(i in 1:length(id.list)){
	
	# read in binary file, get top 100 correlated genes
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
	
	# get the correlation values of interest
	# we are only interested in Entrez ids that have a mapped Refseq id
	corr.of.interest <- correlation[entrez.in.unique.mapping.boo]
	
	o <- order(corr.of.interest, decreasing=T)
	
	# taking 150 instead of 100, because some entrez IDs might still be mapped to the same Refseq
	# of course excluding entry 1 (the input gene itself)
	top.correlated.entrez <- entrez.in.unique.mapping[o[2:151]]
  
	top100.refseqs <- unique(entrez2unique.refseq[as.character(top.correlated.entrez),1])[1:100]
	
	# print output
	outfile <- paste(outfile.dir,"/top_refseqs_",id, ".txt", sep="")
	write.table(file=outfile, top100.refseqs,sep="\t", quote=F, row.names=F, col.names=F)
	
	
}# end foreach ID

