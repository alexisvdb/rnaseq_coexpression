#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include "ringBuffer.h"

void ringBufferInit(ringBufferStruct* ringBuffer,size_t size) {
	ringBuffer->head=0;
	ringBuffer->tail=0;
	ringBuffer->feeder=0;
	ringBuffer->size=size;
	ringBuffer->buffer=calloc(sizeof(void*),size);
	
	assert(ringBuffer->buffer!=NULL);
	pthread_mutex_init(&ringBuffer->lock, NULL);
	pthread_cond_init(&ringBuffer->notFull, NULL);
	pthread_cond_init(&ringBuffer->notEmpty, NULL);
	ringBuffer->label=NULL;
	ringBuffer->readers=0;
}

void ringBufferLabel(ringBufferStruct* ringBuffer,char* label) {
	ringBuffer->label=label;
}
void ringBufferDestroy(ringBufferStruct* ringBuffer) {
	pthread_mutex_destroy(&ringBuffer->lock);
	pthread_cond_destroy(&ringBuffer->notFull);
	pthread_cond_destroy(&ringBuffer->notEmpty);
	/*if(ringBuffer->label!=NULL) {
		free(ringBuffer->label);
	}*/
	free(ringBuffer->buffer);
	//free(ringBuffer);
}

void* ringBufferGet(ringBufferStruct* ringBuffer) {
	pthread_mutex_lock(&ringBuffer->lock);
	if(ringBuffer->feeder == 0) {
		return NULL;
	}
	size_t size=ringBuffer->size;
	while((ringBuffer->tail % size)==(ringBuffer->head % size)) {
		//fprintf(stderr,"%s %i %i empty\n",ringBuffer->label,ringBuffer->head,ringBuffer->tail);
		pthread_cond_wait(&ringBuffer->notEmpty,&ringBuffer->lock);
		//pthread_cond_timedwait(&ringBuffer->notEmpty,&ringBuffer->lock,10);
		//fprintf(stderr,"%s %i %i not empty\n",ringBuffer->label,ringBuffer->head,ringBuffer->tail);
		if(ringBuffer->feeder == 0) {
			pthread_mutex_unlock(&ringBuffer->lock);	
			return NULL;
		}
	}
	void* result=ringBuffer->buffer[ringBuffer->head % size];
	ringBuffer->buffer[ringBuffer->head % size]=NULL;
	ringBuffer->head++;
	pthread_cond_signal(&ringBuffer->notFull);
	pthread_mutex_unlock(&ringBuffer->lock);	
	return result;
}

void ringBufferPut(ringBufferStruct* ringBuffer,void* item) {
	pthread_mutex_lock(&ringBuffer->lock);
	size_t size=ringBuffer->size;
	while(((ringBuffer->tail+1) % size)==(ringBuffer->head % size)) {
		//fprintf(stderr,"%s %i %i full\n",ringBuffer->label,ringBuffer->head,ringBuffer->tail);
		pthread_cond_wait(&ringBuffer->notFull,&ringBuffer->lock);
		//fprintf(stderr,"%s %i %i not full\n",ringBuffer->label,ringBuffer->head,ringBuffer->tail);
	}
	ringBuffer->buffer[ringBuffer->tail % size]=item;
	ringBuffer->tail++;
	pthread_cond_signal(&ringBuffer->notEmpty);
	pthread_mutex_unlock(&ringBuffer->lock);
}

int ringBufferGetFeeder(ringBufferStruct* ringBuffer) {
	pthread_mutex_lock(&ringBuffer->lock);
	int result=ringBuffer->feeder;
	pthread_mutex_unlock(&ringBuffer->lock);
	return result;
}

void ringBufferAddFeeder(ringBufferStruct* ringBuffer) {
	pthread_mutex_lock(&ringBuffer->lock);
	ringBuffer->feeder++;
	pthread_mutex_unlock(&ringBuffer->lock);
}

void ringBufferRemoveFeeder(ringBufferStruct* ringBuffer) {
	pthread_mutex_lock(&ringBuffer->lock);
	ringBuffer->feeder--;
	pthread_cond_broadcast(&ringBuffer->notEmpty);
	pthread_mutex_unlock(&ringBuffer->lock);
	//fprintf(stderr,"%i %s feeders left\n",ringBuffer->feeder,ringBuffer->label);
}

void ringBufferAddReader(ringBufferStruct* ringBuffer) {
	pthread_mutex_lock(&ringBuffer->lock);
	ringBuffer->readers++;
	pthread_mutex_unlock(&ringBuffer->lock);
}

void ringBufferRemoveReader(ringBufferStruct* ringBuffer) {
	pthread_mutex_lock(&ringBuffer->lock);
	ringBuffer->readers--;
	pthread_cond_broadcast(&ringBuffer->notEmpty);
	pthread_mutex_unlock(&ringBuffer->lock);
}	

