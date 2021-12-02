#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include "binIndex.h"
#include "binFormat.h"
int printRecord(scoreRecord record,void* extra) {
	printf("%f\n",record.score);
	return TRUE;
}
int main(int argc, char *argv[]) {
	assert(!isatty(fileno(stdin)));
	walkBinFile(stdin,printRecord,NULL);
	return 
}
