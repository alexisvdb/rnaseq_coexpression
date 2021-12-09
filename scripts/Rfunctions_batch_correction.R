batch_correction = function(dat, method = "none", study.sets, annotation){
  c("none", "removeBatchEffect", "combat", "combat-seq")
  # batch correction
  dat <- switch (method,
    removeBatchEffect = batch_effect_removeBatchEffect(dat, study.sets, annotation),
    combat            = batch_effect_combat(dat, study.sets, annotation),
    combat_seq        = batch_effect_combat_seq(dat, study.sets, annotation),
    none              = dat,
  )
  
  # return result
  dat
  
}



batch_effect_removeBatchEffect = function(dat, study.sets, annotation){
  message("# running removeBatchEffect...")
  library(limma)
  sets <- unique(study.sets[,1])
  result <- NULL # to store final result
  
  for(set in sets){
    message("# processing sample set ",set," out of ",length(sets),"...")
    studies <- rownames(study.sets)[study.sets==set]
    annotation.subset <- subset(annotation, annotation$study_id %in% studies)
    samples <- rownames(annotation.subset)
    
    # get the data for this set
    data.subset <- dat[,samples]
    
    cell.types <- annotation.subset[,2]
    cell.type.n <- length(unique(cell.types))
    
    batches <- annotation.subset[,1]
    batches.n <- length(unique(batches))
    
    subset.size <- length(unique(batches))
    indices <- 1:subset.size
    names(indices) <- unique(batches)
    batch.indices <- indices[batches]
    
    # there are several different cases, requiring different processing:
    # - subset of samples including > 1 cell type
    # - subset of samples of only 1 batch and 1 cell type -> no batch correction needed
    # - subset of samples of > 1 batch but all of the same 1 cell type 
    if(cell.type.n > 1){ # if there is more than 1 cell type in the data
      mod = model.matrix(~as.factor(cell_or_tissue), data=annotation.subset)
      corrected.data = removeBatchEffect(data.subset, batch = batches, design = mod)
    } else if (cell.type.n == 1 & batches.n == 1){ # if there is only 1 cell type in just 1 batch
      corrected.data = data.subset # nothing to be done
    } else if (cell.type.n == 1){ # if there is only 1 cell type in the data
      mod = model.matrix(~1, data=annotation.subset)
      corrected.data = removeBatchEffect(data.subset, batch = batches, design = mod)
    }
    
    # add to result
    if(is.null(result)){
      result <- corrected.data
    } else {
      result <- cbind(result, corrected.data)
    }
    
  }# end for loop running through all sets

  # return the result
  result
}




combat = function(dat, study.sets, annotation){
  message("# running ComBat")
  library(sva)
  sets <- unique(study.sets[,1])
  result <- NULL # to store final result
  
  for(set in sets){
    message("# processing sample set ",set," out of ",length(sets),"...")
    studies <- rownames(study.sets)[study.sets==set]
    annotation.subset <- subset(annotation, annotation$study_id %in% studies)
    samples <- rownames(annotation.subset)
    
    # get the data for this set
    data.subset <- dat[,samples]
    
    cell.types <- annotation.subset[,2]
    cell.type.n <- length(unique(cell.types))
    
    batches <- annotation.subset[,1]
    batches.n <- length(unique(batches))
    
    subset.size <- length(unique(batches))
    indices <- 1:subset.size
    names(indices) <- unique(batches)
    batch.indices <- indices[batches]
    
    # there are several different cases, requiring different processing:
    # - subset of samples including > 1 cell type
    # - subset of samples of only 1 batch and 1 cell type -> no batch correction needed
    # - subset of samples of > 1 batch but all of the same 1 cell type 
    if(cell.type.n > 1){ # if there is more than 1 cell type in the data
      mod = model.matrix(~as.factor(cell_or_tissue), data=annotation.subset)
      corrected.data = ComBat(dat=as.matrix(data.subset), batch=batch.indices, mod=mod, par.prior=TRUE, prior.plots=FALSE)
    } else if (cell.type.n == 1 & batches.n == 1){ # if there is only 1 cell type in just 1 batch
      corrected.data = data.subset # nothing to be done
    } else if (cell.type.n == 1){ # if there is only 1 cell type in the data
      mod = model.matrix(~1, data=annotation.subset)
      corrected.data = ComBat(dat=as.matrix(data.subset), batch=batch.indices, mod=mod, par.prior=TRUE, prior.plots=FALSE)
    }
    
    # add to result
    if(is.null(result)){
      result <- corrected.data
    } else {
      result <- cbind(result, corrected.data)
    }
    
  }# end for loop running through all sets
  
  # return the result
  result
}



batch_effect_combat_seq = function(dat, study.sets, annotation){
  message("# running ComBat-seq")
  library(sva)
  
  # ComBat-seq is different from the other approaches in that it expects
  # integers (read counts) as input. So, let's do a few simple checks to see if
  # the input data here looks OK. Give a warning if not.
  
  # read counts should not be negative
  if(any(dat < 0)){
    warning("# Input data contains negative numbers. 
# ComBat-seq expects read counts as input. 
# Read counts should not be negative")
  }
  
  # check for a few of the input values if they indeed look like integers
  # in R this is a bit complicated because integer data might be type numeric,
  # and not necessarily type integer
  if(any(floor(dat[1:10,1:10]) != dat[1:10,1:10])){
    warning("# Input data does not look like integers. 
# ComBat-seq expects read counts as input.")
  }
  
  sets <- unique(study.sets[,1])
  result <- NULL # to store final result
  
  for(set in sets){
    message("# processing sample set ",set," out of ",length(sets),"...")
    studies <- rownames(study.sets)[study.sets==set]
    annotation.subset <- subset(annotation, annotation$study_id %in% studies)
    samples <- rownames(annotation.subset)
    
    # get the data for this set
    data.subset <- dat[,samples]
    
    cell.types <- annotation.subset[,2]
    cell.type.n <- length(unique(cell.types))
    
    batches <- annotation.subset[,1]
    batches.n <- length(unique(batches))
    
    subset.size <- length(unique(batches))
    indices <- 1:subset.size
    names(indices) <- unique(batches)
    batch.indices <- indices[batches]
    
    # there are several different cases, requiring different processing:
    # - subset of samples including > 1 cell type
    # - subset of samples of only 1 batch and 1 cell type -> no batch correction needed
    # - subset of samples of > 1 batch but all of the same 1 cell type 
    if(cell.type.n > 1){ # if there is more than 1 cell type in the data
      corrected.data = ComBat_seq(data.subset, batch=batches, group=cell.types)
    } else if (cell.type.n == 1 & batches.n == 1){ # if there is only 1 cell type in just 1 batch
      corrected.data = data.subset # nothing to be done
    } else if (cell.type.n == 1){ # if there is only 1 cell type in the data
      corrected.data = ComBat_seq(data.subset, batch=batches, group=NULL)
    }
    
    # add to result
    if(is.null(result)){
      result <- corrected.data
    } else {
      result <- cbind(result, corrected.data)
    }
    
  }# end for loop running through all sets
  
  # return the result
  result
}


