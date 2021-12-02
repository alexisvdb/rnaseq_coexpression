#!/bin/sh

function showUsage {
	echo "$0 inputFile.txt outputFile.bin";
	exit
}

input=$1
output=$2

if [[ -z $input ]] 
then showUsage $0
fi
if [[ -z $output ]] 
then showUsage $0
fi
bin=`dirname $0`
#grep -v "AFFX-" $input | cut -f 2- |tail -n +2 | time -p $bin/massCorrelationGPU -o $output
grep -v "AFFX-" $input | cut -f 2- |tail -n +2 | time -p $bin/massCorrelationThread -o $output
#grep -v "AFFX-" $input | cut -f 2- |tail -n +2 | time -p $bin/massCorrelation -o $output

# this:
# removes the unneeded probes (AFFX)
# removes the probe column (because it only needs the expression values)
# removes the header (=the sample names)
# time measures the time ("time -p" is not really needed)
# and runs the cross-correlation calculation program


# $0 is "this program"
# dirname $0 is the directory in which this program is located

