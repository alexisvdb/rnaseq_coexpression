#!/usr/bin/perl

use strict;

my $q=$ARGV[0];

while(`bjobs -q $q | wc -l` > 0) {
	sleep(10);
}
