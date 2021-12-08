#!/bin/bash

# the directory where the analysis is being run
root=./;

entrez_file=$1;             # ex: temp_split_ncbi/ncbi_00
bin_split_dir=$2;           # ex: binary_files_split_per_gene
all_entrez_list=$3          # ex: ncbi_gene_list.txt
entrez_to_refseq=$4;        # ex: TFBS/entrez2unique_refseqs_human.txt 
output_dir=$5;              # ex: tmp_TFBS_output/
final_output_file=$6;       # ex: tmp_TFBS_output/tfbs_00

present_refseq_file=$7; 	# ex: TFBS/present_refseq_ids_human.txt
gc_cluster_file=$8; 		# ex: TFBS/k2_GC_CpG_clusters_human.txt
tfbs_map_file=$9;			# ex: TFBS/map_tfbs_presence_human.txt
gc_class_stats=${10};		# ex: TFBS/GcClass_stats_human.txt


# some input that should not change between datasets:
pwm_list_file=$root/TFBS/list_pwm_ids.txt
pwm_to_tf_files=$root/TFBS/pwm2tfs.txt

correctionFactor=`cat $root/TFBS/correctionFactor.txt` # contains the number of PWMs used
# this number is used to correct for multiple testing below


rm $final_output_file;
touch $final_output_file;

while read id; do
	echo $id;
	# get the top 100 correlated Refseqs for this entrez ID
	Rscript $root/Rscript_get_top100_refseq_ids.R $id $bin_split_dir $all_entrez_list $entrez_to_refseq $output_dir;
	
	infile=$output_dir/top_refseqs_$id.txt; # the output file of the above command

	# run the motif enrichment analysis on the top 100 refseqs
	$root/bin/enrichment $present_refseq_file $pwm_list_file $gc_cluster_file $infile < $tfbs_map_file | \
	$root/bin/motifCountAddInfo.pl $gc_class_stats $pwm_to_tf_files |\
	$root/bin/motifEnrichmentPvalue $correctionFactor | \
	sort  -k6,6g -k5,5gr | awk '{print NR"\t"$$0}' > $output_dir/enrich_$id.txt;
	
	# process the output
	echo -e -n $id"\t" >>$final_output_file;
	perl $root/TFBS/scan_motif_enrichment_result_file.pl $output_dir/enrich_$id.txt >>$final_output_file;
	
	# clean up
	rm $output_dir/enrich_$id.txt;
	rm $infile;
	
done < $entrez_file


