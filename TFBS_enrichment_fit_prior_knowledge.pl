#!/usr/bin/perl

use strict;
use warnings;

my ( $file_prefix, $map_tfbs_file, $entrez_to_unique_refseq_file ) = @ARGV;

my $ref;


##################################################
### read in prior knowledge
##################################################

# reading refseq to TFBS
$ref = read_gene_to_go_files($map_tfbs_file);
my %refseq2tfbs = %$ref;

$ref = read_gene_to_go_files($entrez_to_unique_refseq_file);
my %entrez2refseq = %$ref;


##################################################
### read in results
##################################################
	
my $counts = 0;
my $counts_total = 0;

my @files = glob($file_prefix."*");

foreach my $file (@files){
	open(FILE, $file) or die $!;
	while(my $line = <FILE>){
		chomp $line;
		my ($entrez, 
			$tfbs_count, $motifs) = split "\t", $line;
		#~ print $line,"\n";
		#~ die;
		
		$counts_total += 1;
		
		if($tfbs_count > 0){
			
			# the the (single) Refseq id assigned to this Entrez ID
			next unless defined $entrez2refseq{$entrez};
			my $refseq = (keys %{$entrez2refseq{$entrez}})[0];
			#~ print$entrez,"\t",$refseq,"\n";
			#~ die;
			
			next unless defined $refseq2tfbs{$refseq};
			
			my @terms = split ";", $motifs;
			my $present_boo = 0;
			foreach my $term (@terms){
				$present_boo = 1 if defined $refseq2tfbs{$refseq} & defined $refseq2tfbs{$refseq}{$term};
				last if $present_boo == 1;
			}
			$counts += 1 if $present_boo == 1;
		}
		
	}# end reading file
	close FILE;
}# end reading all files

print $counts/$counts_total,"\n";


##################################################
##################################################
##################################################
##################################################
##################################################
sub read_gene_to_go_files{
	my ($file) = @_;
	
	my %data = ();
	
	open(FILE, $file) or die $!;
	while(my $line = <FILE>){
		chomp $line;
		
		# skip commented lines
		next if $line =~ /^#/;
		#~ print $line,"\n";
		#~ die;
		my ($id, $go) = split "\t", $line;
		
		$data{$id}{$go} = 1;
		
	}# end reading file
	close FILE;
	
	return \%data;
	
}# end sub read_parameter_file
##################################################

