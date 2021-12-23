#### What does this code do? #### 

# This code reads in the raw quality measures for each data, for each data
# processing workflow. It processes these quality measures to a single quality
# score, and does downstream analysis. It reproduces the figures shown in the
# manuscript.



#### load packages ####
library(ggplot2)
library(gplots) # for heatmap.2



#### read in data ####
file <- "overview_quality_measures.csv"
dat.raw <- read.csv(file)
head(dat.raw)



#### PCA to get a single quality score ####

# the eight raw quality measures
quality.measures.of.interest <- c(
  "go_MF_enrichment", "go_BP_enrichment", "go_CC_enrichment",
  "go_MF_prior", "go_BP_prior", "go_CC_prior",
  "tfbs_enrichment", "tfbs_prior"
  )

dat.pca.input <- dat.raw[,quality.measures.of.interest]
res.pca <- prcomp(dat.pca.input,center = T, scale. = T)

# orientation of the PC axes is arbitrary
# let's make the first axis mostly positive for easier interpretation
if(mean(res.pca$rotation[,1]<0)>.5){
  res.pca$x <- -res.pca$x
  res.pca$rotation <- -res.pca$rotation
}
summary(res.pca)

# make plots of loadings
res.pca$rotation # these are the loadings
loadings <- res.pca$rotation[# reorder columns to facilitate interpretation
  c("go_MF_enrichment", "go_BP_enrichment", "go_CC_enrichment", "tfbs_enrichment",
    "go_MF_prior", "go_BP_prior", "go_CC_prior", "tfbs_prior"
  ),
]
loadings[,1] # loadings for PC1
barplot(loadings[,1])
loadings[,2] # loadings for PC1
barplot(loadings[,2])

# what amount of total variation is explained by each PC?
pr.var=res.pca$sdev^2
pve=pr.var/sum(pr.var)

# Fig. S2A
plot(pve, xlab="Principal Component", ylab="Proportion of Variance Explained", ylim=c(0,1),type='b')

# PC1 is correlated with ALL raw quality measures. We can use it as a general
# quality score, which is easier to interpret than multiple different quality
# measures.

# let's also rescale it to the range 0 to 1
general.quality <- (res.pca$x[,1] - min(res.pca$x[,1])) / (max(res.pca$x[,1]) - min(res.pca$x[,1]))

# make a histogram; Fig. S3AD
ggplot(NULL, aes(general.quality)) + geom_histogram(binwidth = 0.05, colour="white", breaks = seq(-.025,1,.05)) +
  theme(axis.text=element_text(size=12))


# scatterplots of the general score vs the individual measures
# Fig. S2C-D
# for PC1 (equivalent to Quality)
plot(general.quality, dat.pca.input[,1], pch=20, cex=.6)
plot(general.quality, dat.pca.input[,2], pch=20, cex=.6)
plot(general.quality, dat.pca.input[,3], pch=20, cex=.6)
plot(general.quality, dat.pca.input[,4], pch=20, cex=.6)
plot(general.quality, dat.pca.input[,5], pch=20, cex=.6)
plot(general.quality, dat.pca.input[,6], pch=20, cex=.6)
plot(general.quality, dat.pca.input[,7], pch=20, cex=.6)
plot(general.quality, dat.pca.input[,8], pch=20, cex=.6)

# for PC2
plot(res.pca$x[,2], dat.pca.input[,1], pch=20, cex=.6)
plot(res.pca$x[,2], dat.pca.input[,2], pch=20, cex=.6)
plot(res.pca$x[,2], dat.pca.input[,3], pch=20, cex=.6)
plot(res.pca$x[,2], dat.pca.input[,4], pch=20, cex=.6)
plot(res.pca$x[,2], dat.pca.input[,5], pch=20, cex=.6)
plot(res.pca$x[,2], dat.pca.input[,6], pch=20, cex=.6)
plot(res.pca$x[,2], dat.pca.input[,7], pch=20, cex=.6)
plot(res.pca$x[,2], dat.pca.input[,8], pch=20, cex=.6)



# the Pearson correlation between the individual measures and PC1 and PC2
# The correlation values shown in Fig. S2C-D
# with PC1
cor.pc1 <- apply(dat.pca.input, 2, function(x) cor(res.pca$x[,1],x))
# with PC2
cor.pc2 <- apply(dat.pca.input, 2, function(x) cor(res.pca$x[,2],x))


#### prepare data linear regression ####

dat.model <- data.frame(
  "quality"       = general.quality,
  "sample_count"  = dat.raw[,"sample_count"],
  "batch_count"   = dat.raw[,"batch_count"],
  "species"       = factor(dat.raw[,"species"]),
  "normalization" = factor(dat.raw[,"normalization"]),
  "batch"         = factor(dat.raw[,"batch"]),
  "correlation"   = factor(dat.raw[,"correlation"]),
  "cell_index"    = dat.raw[,"cell_index"]
)

# reorder levels so they would be in increasing order in the linear regression analysis below
dat.model$normalization<- factor(dat.model$normalization, levels = c("quantile", "rlog", "cpm", "tmm", "med", "uq", "none"))
dat.model$batch<- factor(dat.model$batch, levels = c("none", "removeBatchEffect", "combat", "combat_seq"))



#### define training and test sets ####
sets.human <- 1:68 # indices of all human cell types and tissues
sets.mouse <- 1:76 # same for mouse

set.seed(1234)
set.human.train <- sort(sample(68,68*3/4)) # select 3/4 of the human cell types and tissues
set.mouse.train <- sort(sample(76,76*3/4)) # same for mouse cell types and tissues

set.human.test <- setdiff(sets.human,set.human.train)
set.mouse.test <- setdiff(sets.mouse,set.mouse.train)

set.human.test
set.mouse.test

# make a backup
dat.model.ALL <- dat.model



#### select just the training cases ####
dat.model <- dat.model[ (dat.model$species == "human" & dat.model$cell_index %in% set.human.train) |
                      (dat.model$species == "mouse" & dat.model$cell_index %in% set.mouse.train)
                    , ]
dim(dat.model)
# should be: 5400 x 8



#### linear regression ####

# everything except combat_seq, which was treated separately in the paper
dat.model.no.combatseq <- subset(dat.model, dat.model$batch!="combat_seq")
res.lm <- lm(quality ~ log10(sample_count) + log10(batch_count) + species + normalization + batch + correlation, 
             data = dat.model.no.combatseq)
summary(res.lm)
# Table 1 in the manuscript
tmp <- summary(res.lm)
tmp$coefficients



#### scatterplot ample count vs quality ####
dat.model <- subset(dat.model, dat.model$batch!="combat_seq")
mean.quality <- c() # to store mean quality of each set
samples <- c()
batches <- c()
for(s in set.human.train){
  subs <- subset(dat.model, cell_index == s & species == "human")
  mean.quality <- c(mean.quality, mean(subs$quality))
  batches <- c(batches, subs$batch_count[1])
  samples <- c(samples, subs$sample_count[1])
}
for(s in set.mouse.train){
  subs <- subset(dat.model, cell_index == s & species == "mouse")
  mean.quality <- c(mean.quality, mean(subs$quality))
  batches <- c(batches, subs$batch_count[1])
  samples <- c(samples, subs$sample_count[1])
}
means <- data.frame(
  samples = samples,
  batches = batches,
  means   = mean.quality,
  species = c(rep("human",length(set.human.train)),rep("mouse",length(set.mouse.train)))
)


# Fig. 3A
p <- ggplot(dat.model, aes(x=sample_count, y=quality)) + 
  geom_point(size=.8, colour="dark grey") + 
  scale_x_log10(breaks=c(20,50,100,200,500,1000, 2000,5000), minor_breaks=NULL) + 
  theme_bw() + 
  theme(legend.position = "none", axis.text=element_text(size=20))
p <- p + geom_point(data=means, aes(x=samples, y=means, color=species), size=3)
p



#### normalization methods ####
dat.model.no.combatseq <- subset(dat.model, dat.model$batch!="combat_seq")

dat.quantile <- subset(dat.model,dat.model$normalization=="quantile")
dat.rlog <- subset(dat.model,dat.model$normalization=="rlog")
dat.cpm <- subset(dat.model,dat.model$normalization=="cpm")
dat.tmm <- subset(dat.model,dat.model$normalization=="tmm")
dat.med <- subset(dat.model,dat.model$normalization=="med")
dat.uq <- subset(dat.model,dat.model$normalization=="uq")

# divide into 3 sets of 36 cell types, according to sample count
dat.model.no.combatseq$size.type <- rep("medium", nrow(dat.model.no.combatseq))
dat.model.no.combatseq$size.type[dat.model.no.combatseq$sample_count < 45] <- "small"
dat.model.no.combatseq$size.type[dat.model.no.combatseq$sample_count >= 112] <- "large"
dat.model.no.combatseq$size.type <- factor(dat.model.no.combatseq$size.type, levels = c("small", "medium", "large"))
summary(dat.model.no.combatseq$sample_count[dat.model.no.combatseq$size.type=="small"])
summary(dat.model.no.combatseq$sample_count[dat.model.no.combatseq$size.type=="medium"])
summary(dat.model.no.combatseq$sample_count[dat.model.no.combatseq$size.type=="large"])
ggplot(dat.model.no.combatseq, aes(x=normalization, y=quality, fill=size.type)) + 
  # geom_violin(position = "dodge") + coord_flip() + 
  geom_boxplot(width=.8, position = "dodge2") +
  theme(axis.text=element_text(size=20), legend.position = "none")
# Fig 3B



#### batch effect treatment ####

dat.model <- dat.model.ALL
# again select just the training data
dat.model <- dat.model[ (dat.model$species == "human" & dat.model$cell_index %in% set.human.train) |
                          (dat.model$species == "mouse" & dat.model$cell_index %in% set.mouse.train)
                        , ]
dim(dat.model)
# should be: 5400 x 8

dat.none <- subset(dat.model,dat.model$batch=="none")
dat.limma <- subset(dat.model,dat.model$batch=="removeBatchEffect")
dat.combat <- subset(dat.model,dat.model$batch=="combat")

# Difference in quality in function of sample counts
# collect data
dat.plot <- data.frame(
  none  = dat.none$quality, 
  removeBatchEffect = dat.limma$quality,
  combat = dat.combat$quality,
  diff_raw_limma = dat.limma$quality-dat.none$quality,
  diff_raw_combat = dat.combat$quality-dat.none$quality,
  sample_count = dat.none$sample_count,
  batch_count = dat.none$batch_count
)

# for removeBatchEffect (limma)
# Fig. 4A
ggplot(dat.plot, aes(x=sample_count, y=diff_raw_limma)) + geom_point(color="dark grey", size=1) + 
  scale_x_log10(breaks=c(20,50,100,200,500,1000, 2000,5000), minor_breaks=NULL) +
  geom_hline(yintercept=0, linetype="dashed", color = "red") + ylim(-0.3,0.4) +
  stat_summary(fun=mean, geom="point", colour="blue", size=3) +
  geom_smooth(method = "loess", se=FALSE, colour="black") + theme_bw() + theme(axis.text=element_text(size=16))


# for combat
# Fig. 4B
ggplot(dat.plot, aes(x=sample_count, y=diff_raw_combat)) + geom_point(color="dark grey", size=1) + 
  scale_x_log10(breaks=c(20,50,100,200,500,1000, 2000,5000), minor_breaks=NULL) +
  geom_hline(yintercept=0, linetype="dashed", color = "red") + ylim(-0.3,0.4) +
  stat_summary(fun=mean, geom="point", colour="blue", size=3) +
  geom_smooth(method = "loess", se=FALSE, colour="black") + theme_bw() + theme(axis.text=element_text(size=16))


# similar, difference in quality in function of batch counts
# Fig. 4D for removeBatchEffect
ggplot(dat.plot, aes(x=batch_count, y=diff_raw_limma)) + geom_point(color="dark grey", size=1) + 
  scale_x_log10(breaks=c(1,2,5,10,20,50,100), minor_breaks=NULL) + 
  geom_hline(yintercept=0, linetype="dashed", color = "red") + ylim(-0.3,0.4) +
  stat_summary(fun=mean, geom="point", colour="blue", size=3) +
  geom_smooth(method = "loess", se=FALSE, colour="black") + theme_bw() + theme(axis.text=element_text(size=16))

# Fig. 4E for ComBat
ggplot(dat.plot, aes(x=batch_count, y=diff_raw_combat)) + geom_point(color="dark grey", size=1) + 
  scale_x_log10(breaks=c(1,2,5,10,20,50,100), minor_breaks=NULL) + 
  geom_hline(yintercept=0, linetype="dashed", color = "red") + ylim(-0.3,0.4) +
  stat_summary(fun=mean, geom="point", colour="blue", size=3) +
  geom_smooth(method = "loess", se=FALSE, colour="black") + theme_bw() + theme(axis.text=element_text(size=16))



#### similar for combatseq ####
dat.none <- subset(dat.model,dat.model$batch=="none")
dat.combatseq <- subset(dat.model,dat.model$batch=="combat_seq" & dat.model$normalization!="none")

dat.plot <- data.frame(
  raw  = dat.none$quality, 
  combatseq = dat.combatseq$quality,
  diff_raw_combatseq = dat.combatseq$quality-dat.none$quality,
  sample_count = dat.none$sample_count,
  batch_count = dat.none$batch_count
)

# Fig. 4C, ComBat-seq
ggplot(dat.plot, aes(x=sample_count, y=diff_raw_combatseq)) + geom_point(color="dark grey", size=1) + 
  scale_x_log10(breaks=c(20,50,100,200,500,1000, 2000,5000), minor_breaks=NULL) +
  geom_hline(yintercept=0, linetype="dashed", color = "red") + ylim(-0.4,0.4) +
  stat_summary(fun=mean, geom="point", colour="blue", size=3) +
  geom_smooth(method = "loess", se=FALSE, colour="black") + theme_bw() + theme(axis.text=element_text(size=16))


# Fig. 4F
ggplot(dat.plot, aes(x=batch_count, y=diff_raw_combatseq)) + geom_point(color="dark grey", size=1) + 
  scale_x_log10(breaks=c(1,2,5,10,20,50,100), minor_breaks=NULL) + 
  geom_hline(yintercept=0, linetype="dashed", color = "red") + ylim(-0.4,0.4) +
  stat_summary(fun=mean, geom="point", colour="blue", size=3) +
  geom_smooth(method = "loess", se=FALSE, colour="black") + theme_bw() + theme(axis.text=element_text(size=16))



#### correlation measure ####
dat.pearson <- subset(dat.model,dat.model$correlation=="pearson")
dat.spearman <- subset(dat.model,dat.model$correlation=="spearman")

dat.plot <- data.frame(
  diff         = dat.pearson$quality - dat.spearman$quality,
  sample_count = dat.pearson$sample_count
)

# Fig. 5
ggplot(dat.plot, aes(x=sample_count, y=diff)) + geom_point(color="dark grey", size=1) + 
  scale_x_log10(breaks=c(20,50,100,200,500,1000, 2000,5000), minor_breaks=NULL) +
  geom_hline(yintercept=0, linetype="dashed", color = "red") + ylim(-0.4,0.3) +
  stat_summary(fun=mean, geom="point", colour="blue", size=3) +
  geom_smooth(method = "loess", se=FALSE, colour="black") + theme_bw() + theme(axis.text=element_text(size=16))



#### evaluate on test data ####
# we want to see if the "optimized" data processing workflow is better than
# a "control" work flow
dat.raw.test <- dat.raw[ (dat.raw$species == "human" & dat.raw$cell_index %in% set.human.test) |
                      (dat.raw$species == "mouse" & dat.raw$cell_index %in% set.mouse.test)
                    , ]
dim(dat.raw.test)
# 1800 x 15

# as "control" processing flow, take:
# - rlog normalization (as used in Deseq2) 
# - no batch correction
# - pearson correlation
control.measure <- subset(dat.raw.test, normalization=="rlog" & batch=="none" & correlation=="pearson")

# as "optimized" processing flow, take:
# - Upper Quartile normalizaton ("uq")
# - batch effect correction using ComBat
# - spearman correlation for smaller datasets (<= 30 samples), or else pearson correlation 
optimized.measure1 <- subset(dat.raw.test, normalization=="uq" & batch=="combat" & correlation=="spearman" & sample_count <= 30)
optimized.measure2 <- subset(dat.raw.test, normalization=="uq" & batch=="combat" & correlation=="pearson" & sample_count > 30)
optimized.measure <- rbind(optimized.measure1, optimized.measure2)

# need to be put in the same order!
o <- order(control.measure$species, control.measure$cell_index)
control.measure <- control.measure[o,quality.measures.of.interest]

o <- order(optimized.measure$species, optimized.measure$cell_index)
optimized.measure <- optimized.measure[o,quality.measures.of.interest]

t.test.p <- rep(NA,length(quality.measures.of.interest))
names(t.test.p) <- quality.measures.of.interest
cases.improved <- t.test.p

# run through the quality measures, make figures, perform t-test
# these are Fig. 6
for(i in 1:length(quality.measures.of.interest)){
  measure <- quality.measures.of.interest[i]
  df <- data.frame(
    type = factor(c(rep("control",nrow(control.measure)),rep("optimized",nrow(optimized.measure)))),
    value = c(control.measure[,i],optimized.measure[,i])
  )
  ggplot(df, aes(x=type, y=value, fill=type)) + 
    geom_dotplot(binaxis = 'y', stackdir = 'center', stackratio = .9, dotsize = 1, binpositions = "all") +
    theme(legend.position = "none", axis.text=element_text(size=25))
  filename <- paste0("dotplot_control_vs_optimized_",measure,".pdf")
  ggsave(file=filename, width = 6, height = 5)
  t.test.p[i] <- t.test(control.measure[,i],optimized.measure[,i], alternative = "less", paired = TRUE)$p.value
  cases.improved[i] <- sum(control.measure[,i]<optimized.measure[,i]) # fraction of datasets with improvement
}
t.test.p
# the p values shown in Fig. 6

cases.improved
# the number of datasets in which an improvement was found (out of 36)



#### heatmap overview of all methods on all datasets #####

datasets <- c(paste("human", sets.human, sep = "_"),paste("mouse", sets.mouse, sep = "_"))

all.normalizations <- c("rlog", "cpm", "tmm", "uq", "med", "quantile")
all.batch          <- c("none", "combat", "removeBatchEffect", "combat_seq")
all.correlation    <- c("pearson", "spearman")

tmp <- as.matrix(expand.grid(all.normalizations, all.batch, all.correlation))
tmp <- rbind(tmp, c("none", "combat_seq", "pearson")) #add manually
tmp <- rbind(tmp, c("none", "combat_seq", "spearman")) #add manually
combis <- apply(tmp,1,function(x) paste(x,collapse="_"))

overview <- matrix(NA, nrow=length(datasets), ncol=50)
colnames(overview) <- combis
rownames(overview) <- datasets

sizes <- rep(NA, length(datasets))
names(sizes) <- datasets
# run through all methods on all datasets and collect quality scores
dat.model <- dat.model.ALL
for(i in 1:nrow(dat.model)){
  dataset <- paste(dat.model$species[i],"_",dat.model$cell_index[i], sep="")
  
  method <- dat.model[i,c("normalization","batch","correlation")]
  method <- paste(as.matrix(method),collapse = "_")
  overview[dataset,method] <- dat.model[i,"quality"]
  
  sizes[dataset] <- dat.model$sample_count[i]
}

o.size <- order(sizes)

# let's give a color to each method for each dataset
cols <- matrix("white", nrow=nrow(overview), ncol=ncol(overview))
colnames(cols) <- combis
rownames(cols) <- datasets
pal <- colorRampPalette(c("dark blue", "white", "dark red"))(10)
for(d in 1:nrow(overview)){
  tmp <- overview[d,]
  o.tmp <- order(tmp, decreasing = TRUE)
  m <- mean(tmp)
  cols[d,tmp < m]         <- pal[7]
  cols[d,rev(o.tmp)[1]] <- pal[10]
  cols[d,rev(o.tmp)[2]] <- pal[9]
  cols[d,rev(o.tmp)[3]] <- pal[8]
  
  cols[d,tmp >= m]        <- pal[4]
  cols[d,o.tmp[1]] <- pal[1]
  cols[d,o.tmp[2]] <- pal[2]
  cols[d,o.tmp[3]] <- pal[3]
  
}
x <- matrix(1:prod(dim(overview)),nrow=nrow(overview))
colnames(x) <- combis
rownames(x) <- datasets

# sort methods by their mean quality
mean.quality <- apply(overview,2,mean)
o.q <- order(mean.quality, decreasing = TRUE)

heatmap.2(t(x[rev(o.size),o.q]), 
          Rowv=FALSE, Colv=FALSE,dendrogram = "none", 
          scale = "none", col=cols, trace = "none",
          key = FALSE, lhei=c(.2,1), lwid = c(.05,1))
# Fig S4



#### barplot in Fig. 2 ####
datasets.train <- c(paste("human", set.human.train, sep = "_"),
                    paste("mouse", set.mouse.train, sep = "_"))
overview.train <- overview[datasets.train,]

mean.quality <- apply(overview.train,2,mean)
o.q <- order(mean.quality, decreasing = TRUE)

df <- data.frame(
  x = factor(combis[o.q], levels=rev(combis[o.q])),
  mean.q = mean.quality[o.q]
)
# this is the barplot in Fig. 2
ggplot(df, aes(y=x, x=mean.q)) + coord_cartesian(xlim = c(0.39, .53)) +
  geom_bar(stat = "identity") + theme_bw() 

# in how many datasets is a combination better than average?
mean.quality.of.dataset <- apply(overview.train,1,mean)
above.average <- apply(overview.train > mean.quality.of.dataset,2,sum)
above.average[o.q]
# numbers in Fig. 2 showing number of networks above average quality


