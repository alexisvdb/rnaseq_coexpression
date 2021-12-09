#!/usr/bin/perl

use strict;
use warnings;

my ( $file_prefix, $go_MF_file, $go_BP_file, $go_CC_file ) = @ARGV;

my $ref;


##################################################
### read in prior knowledge
##################################################

$ref = read_gene_to_go_files($go_MF_file);
my %go_MF = %$ref;

$ref = read_gene_to_go_files($go_BP_file);
my %go_BP = %$ref;

$ref = read_gene_to_go_files($go_CC_file);
my %go_CC = %$ref;


##################################################
### read in results
##################################################

my $counts_MF = 0;
my $counts_BP = 0;
my $counts_CC = 0;
my $counts_total = 0;

my @files = glob($file_prefix."*");

foreach my $file (@files){
	open(FILE, $file) or die $!;
	while(my $line = <FILE>){
		chomp $line;
		next if $line =~ /^MF_count/;
		my ($entrez, 
			$mf_count, $mf_terms, 
			$bp_count, $bp_terms, 
			$cc_count, $cc_terms) = split "\t", $line;
		#~ print $line,"\n";
		#~ die;
		
		$counts_total += 1;
		
		if($mf_count > 0){
			my @terms = split ";", $mf_terms;
			my $present_boo = 0;
			foreach my $term (@terms){
				$present_boo = 1 if defined $go_MF{$entrez}{$term};
				last if $present_boo == 1;
			}
			$counts_MF += 1 if $present_boo == 1;
		}
		
		if($bp_count > 0){
			my @terms = split ";", $bp_terms;
			my $present_boo = 0;
			foreach my $term (@terms){
				$present_boo = 1 if defined $go_BP{$entrez}{$term};
				last if $present_boo == 1;
			}
			$counts_BP += 1 if $present_boo == 1;
		}
		
		if($cc_count > 0){
			my @terms = split ";", $cc_terms;
			my $present_boo = 0;
			foreach my $term (@terms){
				$present_boo = 1 if defined $go_CC{$entrez}{$term};
				last if $present_boo == 1;
			}
			$counts_CC += 1 if $present_boo == 1;
		}
	}# end reading file
	close FILE;
}# end reading all files

print $counts_MF/$counts_total, "\t", $counts_BP/$counts_total, "\t", $counts_CC/$counts_total,"\n";


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
		
		my ($entrez, $go) = split "\t", $line;
		
		$data{$entrez}{$go} = 1;
		
	}# end reading file
	close FILE;
	
	return \%data;
	
}# end sub read_parameter_file
##################################################

