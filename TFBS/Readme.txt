various_tfbs_data.tar.gz contains the following files:

A mapping between Entrez and Refseq ids, for both human and mouse genes
- entrez2unique_refseqs_human.txt
- entrez2unique_refseqs_mouse.txt

# a list of PWM motifs used in the analysis
- list_pwm_ids.txt

# a file with the number of PWM motifs used in the analysis
# this number is used to correct for multiple testing
- correctionFactor.txt

A mapping between Refseq ids and PWM motifs that are predicted to have hits in their promoter sequence
- map_tfbs_presence_human.txt
- map_tfbs_presence_mouse.txt

A list of Refseq ids present in the data, for both human and mouse
- present_refseq_ids_human.txt
- present_refseq_ids_mouse.txt

# Data about the classification of promoters into 2 classes, one with high GC content 
# and one with low GC content. GcClass_stats show the frequency of PWM hits in each class.
# This data was originally produced for the study Vandenbon et al., PNAS, 2016.
- GcClass_stats_human.txt
- GcClass_stats_mouse.txt
- k2_GC_CpG_clusters_human.txt
- k2_GC_CpG_clusters_mouse.txt
