#include "binFormat.h"
#include <sys/types.h>
#include <unistd.h>
#include <stdlib.h>
#include <assert.h>
#include <fcntl.h>



void walkBinFileQueryFd(int input,index_t query,int indexCount,walkBinFileCallback callback,void* extra) {
	scoreRecord record; // score record storage
	const size_t size=sizeof(scoreRecord); // this size is not what I want
	long offset=0;
	lseek64(input,0,SEEK_SET);
	void** params=(void**)extra;
	score_t *data=(score_t*)params[0];
	if(query>0) {
		for(int i=0;i<query;i++) {
			int local=query-1-i*2;
			off64_t address=(offset+local)*size;
			assert(pread64(input,&record,size,address)==size);
			// assert((record.a==query)||(record.b==query));
			//if(!callback(record,extra)) {
			//	return;
			//}
			data[i]=record.score;
			offset+=indexCount-i;
		}
		off64_t address=(offset-query)*size;
		lseek64(input,address,SEEK_SET);
	}
	
	// correlation with itself
	data[query]=1.0;
	
	int count = indexCount - (query +1);
	scoreRecord buffer[count];
	assert(read(input,buffer,size*count)==size*count);
	for(int i=0;i<count;i++) {
		record=buffer[i];
		//printf("%i %i\n",record.a,record.b);
		// assert((record.a==query)||(record.b==query));
		//if(!callback(record,extra)) {
		//	return;
		//}		
		data[i+query+1]=record.score;
	}
	/*
	for(int i=query+1;i<indexCount;i++) {
		assert(fread(&record,sizeof(scoreRecord),1,input)==1);
		if(!callback(record,extra)) {
			return;
		}
	}*/
}

void walkBinFileQuery(FILE* input,index_t query,int indexCount,walkBinFileCallback callback,void* extra) {
	size_t size=sizeof(scoreRecord);
	long offset=0;
	
	scoreRecord record;
	if(query>0) {
		for(int i=0;i<query;i++) {
			int local=query-1-i*2;
			off_t address=offset+local;	
			assert(fseeko(input,address*size,SEEK_SET)==0);
			//fprintf(stderr,"request : %ld \toffset : %ld \t %i/%i \n",address*size,ftell(input),i,query);

			int count=fread(&record,sizeof(scoreRecord),1,input);
			if(count!=1) {
				fprintf(stderr,"%i %i %i\n",count,ferror(input),feof(input));
				abort();
			}
			// assert((record.a==query)||(record.b==query));
			if(!callback(record,extra)) {
				return;
			}
			offset+=indexCount-i;
		}
		fseek(input,(offset-query)*size,SEEK_SET);
	}
	//fputs("pass\n",stderr);
	for(int i=query+1;i<indexCount;i++) {
		assert(fread(&record,sizeof(scoreRecord),1,input)==1);
		if(!callback(record,extra)) {
			return;
		}
	}

}

void walkBinFile(FILE* input,walkBinFileCallback callback,void* extra) {
	scoreRecord record;
	while(!feof(input) && (fread(&record,sizeof(scoreRecord),1,input)==1)) {
		if(!callback(record,extra)) {
			return;
		}
	}
}

