#include <stdio.h>
#include <stdlib.h>

int main(int argc,char** argv) {
	float delta=atof(argv[1]);
	int range=(2.0/delta) + 1;
	int counts[range+1];
	for(int i=0;i<range;i++) {
		counts[i]=0;
	}

	float value=0;
	while(fread(&value,sizeof(float),1,stdin)==1) {
		if((value>=-1.0) && (value<=1.0)) {
			int position=(value+1.0)/delta;
			counts[position]++;
		}else{
			// fprintf(stderr, "invalid value %0.6f\n", value);
			// abort();
		}
	}
	for(int i=0;i<range;i++) {
		fprintf(stdout, "%i\t%0.6f\t%i\n", i,delta*i-1.0,counts[i]);
	}
}
