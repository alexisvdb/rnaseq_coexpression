#!/usr/local/bin/perl -w
#$ -S /usr/local/bin/perl
#$ -cwd
use strict;
use warnings;
use POSIX;
use Storable qw(dclone);
$|=1;

######################################################################
##### INPUT PARAMETERS
######################################################################

my ($goslim_mapping_file, $goslim_file, $go_annotation_file, $symbol2entrez_file, $entrez_list_file) = @ARGV;
# $goslim_mapping_file is the mapping of each GO term all its (direct and indirect) parent terms
# $goslim_fileshould be the GO .obo file
# $go_annotation_file is the file with the GO annotation data for the genes of an organism
# Note: although the parameter names say "goslim" the data does not necessarilty have to be GO Slim terms

my $ref;
my %goslim;
my %go_mapping;
my %annotations;

########################################
### step 1: read in the goslim terms
########################################
# read in the ids, names, namespaces, and definition

$ref = read_goslim($goslim_file, 1);
%goslim = %$ref;

########################################
### step 2: read mapping of GO terms to GO slim terms
########################################
# read in the GO ids and the GO slim id(s) which they have been mapped to 

$ref = read_mapping($goslim_mapping_file);
%go_mapping = %$ref;

########################################
### step 3: read GO annotations of a set of genes
########################################
# read in the ids, names, namespaces, and definition

$ref = read_annotations($go_annotation_file);
%annotations = %$ref;

########################################
### step 4: read in symbols to entrez ids
########################################

$ref = read_symbol2entrez($symbol2entrez_file);
my %symbol2entrez = %$ref;

$ref = read_list($entrez_list_file);
my %present_entrez_list = %$ref;


########################################
### step 5: process annotations to GO slim annotations and print them
########################################

foreach my $gene (keys %annotations){
	#~ print $gene,"\n";
	
	my %go_slims = ();
	
	foreach my $go (keys %{$annotations{$gene}}){
		# get GO slim term (if exists)
		if(defined $go_mapping{$go}){
			foreach my $goslim (keys %{$go_mapping{$go}}){
				# add any go slims to the set of go slims associated with this gene
				$go_slims{$goslim} = $annotations{$gene}{$go};
			}
		}
		
	}
	
	# print this for all Entrez IDs of this gene (though there should typically be only 1 I guess)
	# if there is no Entrez ID: skip
	foreach my $entrez (keys %{$symbol2entrez{$gene}}){
		# only do these things for Entrez IDs that are actually in the list
		next unless defined $present_entrez_list{$entrez};
		
		# print all remaining GO slim annotations for this gene
		foreach my $goslim (keys %go_slims){
			my $class = $go_slims{$goslim};
			print $entrez,"\t", $goslim,"\t",$class,"\n";
		}
	}
	
}# end foreach gene

####################################################
####################################################
####################################################
####################################################
####################################################
sub read_annotations{
	my ($file) = @_;
	
	my %result = ();
	
	open(FILE, $file) or die $!;
	while(my $line = <FILE>){
		chomp $line;
		next if $line =~ /^!/;
		next unless $line =~ /\tGO\:\d+/;
		my @temp = split "\t", $line;
		
		my $gene 			= $temp[2];
		my $go 				= $temp[4];
		my $annotation_code 	= $temp[6];
		my $annotation_class 	= $temp[8];
		
		$result{$gene}{$go} = $annotation_class;
		
	}# end reading file
	close FILE;
	
	return \%result;
	
}# end sub read_annotations
####################################################
sub read_mapping{
	my ($file) = @_;
	
	my %result = ();
	
	open(FILE, $file) or die $!;
	while(my $line = <FILE>){
		chomp $line;
		next unless $line =~ /(.+)\t(.+)/;
		my $go 		= $1;
		my $go_slim 	= $2;
		
		$result{$go}{$go_slim} = 1;
		
	}# end reading file
	close FILE;
	return \%result;
}# end sub read_mapping
####################################################
sub read_goslim{
	
	my ($file, $skip_root_nodes) = @_;
	
	my %result = ();
	my $counter = 0;
	
	open(FILE, $file) or die $!;
	while(my $line = <FILE>){
		
		# read until you get to the GO term definitions
		next unless $line =~ /^id\: (GO\:\d+)/;
		my $id 			= $1;
		
		# exclude the 3 root nodes
		next if $skip_root_nodes eq "1" and $id eq "GO:0008150"; # biological_process
		next if $skip_root_nodes eq "1" and $id eq "GO:0003674"; # molecular_function
		next if $skip_root_nodes eq "1" and $id eq "GO:0005575"; # cellular_component
		
		
		$result{$id}{name} 		= "";
		$result{$id}{namespace} 	= "";
		$result{$id}{def}			= "";
		$counter += 1;
		
		while(my $line = <FILE>){
			chomp $line;
			
			# go out of this part if you encounter a new term annotation
			last if $line =~ /^\[Term\]/;
			
			$result{$id}{name} = $1 if $line =~ /^name\: (.+)/;
			$result{$id}{namespace} = $1 if $line =~ /^namespace\: (.+)/;
			$result{$id}{name} = $1 if $line =~ /^def\: (.+)/;
			
			
		}
	}# end reading file
	close FILE;
	
	print STDERR "Read in $counter GO ids and definitions.\n";
	
	return \%result;
}# end sub read_goslim
####################################################
sub read_symbol2entrez{
	
	my ($file) = @_;
	
	my %result = ();
	
	open(FILE, $file) or die $!;
	while(my $line = <FILE>){
		chomp $line;
		next if $line =~ /^#/;
		next unless $line =~ /.+\t.+/;
		my @temp = split "\t", $line;
		
		my $symbol 			= $temp[0];
		my $entrez 			= $temp[1];
		
		$result{$symbol}{$entrez} = 1;
		
	}# end reading file
	close FILE;
	
	return \%result; 
	
}# end sub read_symbol2entrez
####################################################
sub read_list{
	my ($file) = @_;
	
	my %result = ();
	
	open(FILE, $file) or die $!;
	while(my $line = <FILE>){
		chomp $line;
		next if $line =~ /^#/;
		
		my $entry 	= $line;
		$result{$entry} = 1;
		
	}# end reading file
	close FILE;
	
	return \%result; 
	
} # end
####################################################
