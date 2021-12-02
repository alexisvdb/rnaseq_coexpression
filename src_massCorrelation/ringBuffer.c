#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <pthread.h>
#include <stdbool.h>
#include "ringBuffer.h"

struct rb_ringBuffer {
	void** buffer;
	size_t capacity;
	volatile size_t head;
	volatile size_t tail;
	pthread_mutex_t lock;
	pthread_cond_t notFull;
	pthread_cond_t notEmpty;
};

rb_ringBuffer_t* rb_init(size_t capacity) {
	rb_ringBuffer_t *result=malloc(sizeof(rb_ringBuffer_t));
	assert(result!=NULL);
	result->buffer=calloc(capacity,sizeof(void*));
	assert(result->buffer!=NULL);
	result->head=0;
	result->tail=0;
	result->capacity=capacity;

	result->lock=(pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
	result->notFull=(pthread_cond_t)PTHREAD_COND_INITIALIZER;
	result->notEmpty=(pthread_cond_t)PTHREAD_COND_INITIALIZER;

	return(result);
}

void rb_destroy(rb_ringBuffer_t* buffer) {
	if(rb_count(buffer)>0) {
		// destroy remaining queue element ? 
		// I think it would be better to warn the developper of this condition
		// as we don't know how to destroy the stored element
		fprintf(stderr,"non-empty queue destroyed ( %zu element(s) remaining )\n",rb_count(buffer));
	}
	free(buffer->buffer);
	free(buffer);
}

void rb_enqueue(rb_ringBuffer_t* buffer,void* item) {
	pthread_mutex_lock(&(buffer->lock));
	while(rb_isFull(buffer)) {
		pthread_cond_wait(&(buffer->notFull),&(buffer->lock));
	}
	buffer->buffer[buffer->tail%buffer->capacity] = item;
	buffer->tail++;
	pthread_mutex_unlock(&(buffer->lock));
	pthread_cond_broadcast(&(buffer->notEmpty));
}

void* rb_dequeue(rb_ringBuffer_t* buffer) {
	pthread_mutex_lock(&(buffer->lock));
	while(rb_count(buffer)==0) {
		pthread_cond_wait(&(buffer->notEmpty),&(buffer->lock));
	}
	void* result=buffer->buffer[buffer->head%buffer->capacity];
	buffer->head++;
	pthread_mutex_unlock(&(buffer->lock));
	pthread_cond_broadcast(&(buffer->notFull));
	return result;
}

size_t rb_count(rb_ringBuffer_t* buffer) {
	return buffer->tail-buffer->head;
}

bool rb_isFull(rb_ringBuffer_t* buffer) {
	return rb_count(buffer)==(buffer->capacity-1);
}

bool rb_enqueueNonBlocking(rb_ringBuffer_t* buffer,void* item) {
	bool result=false;
	pthread_mutex_lock(&(buffer->lock));
	if(!rb_isFull(buffer)) {
		buffer->buffer[buffer->tail%buffer->capacity] = item;
		buffer->tail++;
		result=true;
	}
	pthread_mutex_unlock(&(buffer->lock));
	pthread_cond_broadcast(&(buffer->notEmpty));
	return(result);
}

bool rb_dequeueNonBlocking(rb_ringBuffer_t* buffer,void** output) {
	bool result=false;
	pthread_mutex_lock(&(buffer->lock));
	if(rb_count(buffer)>0) {
		*output=buffer->buffer[buffer->head%buffer->capacity];
		buffer->head++;
		result=true;
	}
	pthread_mutex_unlock(&(buffer->lock));
	pthread_cond_broadcast(&(buffer->notFull));
	return(result);
}

