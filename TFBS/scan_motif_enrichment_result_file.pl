#!/usr/bin/perl

#####################################################################

my ( $result_file ) = @ARGV;

my $p_threshold = 0.01;
my @enriched_pwm = ();

# open file and read in results
open(FILE, $result_file) or die $!;
while(my $line = <FILE>){
	chomp $line;
	
	my @temp = split "\t", $line;
	my $pwm 	= $temp[1];
	my $p		= $temp[6];
	
	# only for the significantly enriched ones
	next unless $p <= $p_threshold;
	
	push @enriched_pwm, $pwm;
	
}# end reading file
close FILE;

# print results
print int(@enriched_pwm),"\t",join(";",@enriched_pwm),"\n";