#pragma once

//#define _LARGEFILE64_SOURCE
#define _FILE_OFFSET_BITS 64
#include <stdio.h>
#include <stdint.h>

#ifndef TRUE
#define TRUE	(1)
#define FALSE	(0)
#endif

typedef uint16_t index_t; // data type for index position
typedef float score_t; // data type for score in storage

typedef struct {
	//index_t a; // index of probe 1
	//index_t b; // index of probe 2
	score_t score; // score 
} scoreRecord;

/* 
callback for walking the binary file, user provided function
@param : score record 
@param : user provided parameter
*/
typedef int (walkBinFileCallback)(scoreRecord,void* extra);

/*
walk binary file for correlation storage and call user provided function
@param : input file to read from 
@param : query : probe id to find 
@param : indexCount : total count of probes
@param : callback : user provided function
@param : user parameter for callback  
*/
void walkBinFileQuery(FILE* input,index_t query,int indexCount,walkBinFileCallback callback,void* extra);

/*
walk binary file for correlation storage and call user provided function, test with file descriptor
@param : input file to read from 
@param : query : probe id to find 
@param : indexCount : total count of probes
@param : callback : user provided function
@param : user parameter for callback  
*/

void walkBinFileQueryFd(int input,index_t query,int indexCount,walkBinFileCallback callback,void* extra);


/*
walk binary file for correlation storage and call user provided function, 
@param : input file to read from
@param : user function to call on each score entry
@param : param for the user function
*/
void walkBinFile(FILE* input,walkBinFileCallback callback,void* extra);

