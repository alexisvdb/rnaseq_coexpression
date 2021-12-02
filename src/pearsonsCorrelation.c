#include <stdio.h>
#include <math.h>
#include "pearsonsCorrelation.h"

extern double getPearsonsCorrelation(const int length,const double* x,const double* y) {

	double sum_x=0.0L;
	double sum_y=0.0L;
	double sum_xy=0.0L;
	double sum_x2=0.0L;
	double sum_y2=0.0L;

	for(int i=0;i<length;i++) {
		double a=x[i];
		double b=y[i];
		sum_x += a;
		sum_y += b;
		sum_xy += a*b;
		sum_x2 += a*a;
		sum_y2 += b*b;

	}
	
	double s1= (sum_xy*length) - (sum_x*sum_y);
	double s2= (sum_x2*length) - (sum_x*sum_x);
	double s3= (sum_y2*length) - (sum_y*sum_y);
	double s4= sqrt(s2*s3);
	double result=s1/s4;
	//fprintf(stderr,"%f %f %f %f %f\n",result,s1,s2,s3,s4);
	//fprintf(stderr,"%f %f %f %f %f\n",sum_x,sum_y,sum_x2,sum_y2,sum_xy);

	return(result);
}

