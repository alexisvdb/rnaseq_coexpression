normalize = function(dat, method = "none"){
  
  # normalize
  dat <- switch (method,
    quantile = normalization_quantile(dat), 
    rlog     = normalization_rlog(dat),
    cpm      = normalization_cpm(dat),
    tmm      = normalization_tmm(dat),
    med      = normalization_med(dat),
    uq       = normalization_uq(dat),
    none     = dat,
  )
  
  # set a small pseudo count
  pseudo <- quantile(dat[dat > 0], 0.01)
  
  # and convert to log10 values
  dat.log10 <- log10(dat+(10*pseudo))
  
  # return result
  dat.log10
  
}



normalization_quantile = function(dat){
  message("# running quantile normalization...")
  library("limma")
  normalizeQuantiles(dat)
}



normalization_rlog = function(dat){
  message("# running rlog (DESeq2) normalization...")
  library("DESeq2")
  
  # remove ridiculously large read counts in the ComBat-seq output (some >>> 1e30!!)
  # DESeq2 can not handle them.
  dat[dat>1e9] <- 1e9
  
  coldata <- matrix(NA, nrow=ncol(dat), ncol=1)
  dds <- DESeqDataSetFromMatrix(countData = dat,
                                colData = coldata,
                                design= ~ 1)
  dds <- estimateSizeFactors(dds) # run this or DESeq() first
  # return the normalized tag counts
  counts(dds, normalized=T)
} 



normalization_cpm = function(dat){
  message("# running cpm normalization...")
  total_tag_count <- apply(dat,2,sum)
  for(c in 1:ncol(dat)){
    dat[,c] <- dat[,c]*1e6/total_tag_count[c]
  }
  # return result
  dat
}



normalization_tmm = function(dat){
  message("# running tmm normalization...")
  library("edgeR")
  # make a DGEList object from the counts
  dat <- DGEList(counts=dat)
  
  # normalization 
  # by default this is the TMM (trimmed mean of M-values)
  dat <- calcNormFactors(dat)
  cpm(dat)
}



normalization_med = function(dat){
  library("edgeR")
  
  message("# running med normalization...")
  # get the median of the NON-ZERO values
  meds <- apply(dat, 2, function(x) median(x[x!=0]))
  mean.med <- exp(mean(log(meds))) # the geometric mean
  correction.factors <- mean.med/meds
  
  # make a DGEList object from the counts
  dat <- DGEList(counts=dat)
  dat$samples[,3] <- correction.factors
  
  # return the normalized tag counts
  cpm(dat)
}



normalization_uq = function(dat){
  library("edgeR")
  
  # UQ gives troubles if too many genes have 0 values in all samples
  # to avoid problems, I adjusted the original approach, and implemented 
  # a few steps by myself
  
  message("# running uq normalization...")
  # get the UQ of the NON-ZERO values
  uqs <- apply(dat, 2, function(x) quantile(x[x!=0],0.75))
  mean.uq <- exp(mean(log(uqs))) # the geometric mean
  correction.factors <- mean.uq/uqs
  
  # make a DGEList object from the counts
  dat <- DGEList(counts=dat)
  dat$samples[,3] <- correction.factors
  
  # return the normalized tag counts
  cpm(dat)
}





