#pragma once
#include <pthread.h>

typedef struct {
	void** buffer;
	int head;
	int tail;
	size_t size;
	pthread_mutex_t lock;
	pthread_cond_t notFull;
	pthread_cond_t notEmpty;
	int feeder;
	char* label;
	int readers;
}  ringBufferStruct;

void ringBufferInit(ringBufferStruct* ringBuffer,size_t size);
void ringBufferLabel(ringBufferStruct* ringBuffer,char* label);
void* ringBufferGet(ringBufferStruct* ringBuffer);
void ringBufferPut(ringBufferStruct* ringBuffer,void* item);
int ringBufferGetFeeder(ringBufferStruct* ringBuffer);
void ringBufferAddFeeder(ringBufferStruct* ringBuffer);
void ringBufferRemoveFeeder(ringBufferStruct* ringBuffer);
void ringBufferAddReader(ringBufferStruct* ringBuffer);
void ringBufferRemoveReader(ringBufferStruct* ringBuffer);
void ringBufferDestroy(ringBufferStruct* ringBuffer);