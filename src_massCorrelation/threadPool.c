#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <unistd.h>
#include <stdbool.h>
#include "threadPool.h"
#include "ringBuffer.h"

typedef struct tp_task {
	tp_worker_t* func;
	void* arg;
} tp_task_t;

struct tp_threadpool {
	unsigned int count;
	volatile bool destroying;
	pthread_mutex_t lock;
	volatile int active;
	pthread_t* workers;
	rb_ringBuffer_t *jobQueue;
};

static void tp_setActive(tp_threadpool_t *pool,int modifier) {
	pthread_mutex_lock(&(pool->lock));
	pool->active+=modifier;
	pthread_mutex_unlock(&(pool->lock));
}

static void* tp_workerThread(void* arg) {
	tp_threadpool_t* pool=arg;
	while(!pool->destroying) {
		tp_task_t *task=NULL;
		if(rb_dequeueNonBlocking(pool->jobQueue,&task)) {
			//=rb_dequeue(pool->jobQueue);
			//fprintf(stderr,"%02x : working \n",pthread_self());
			tp_setActive(pool,1);
			task->func(task->arg);
			free(task);
			tp_setActive(pool,-1);
		}else{
			pthread_yield();
			//fprintf(stderr,"%02x : empty queue, pausing\n",pthread_self());
			//usleep(1000);
		}
	}
	return NULL;
}

extern tp_threadpool_t* tp_init(unsigned int count) {
	tp_threadpool_t *result=malloc(sizeof(tp_threadpool_t));
	result->count=count;
	result->destroying=false;
	result->jobQueue=rb_init(count*10);
	result->lock=(pthread_mutex_t)PTHREAD_MUTEX_INITIALIZER;
	result->active=0;
	result->workers=calloc(count,sizeof(pthread_t));
	for(unsigned int i=0;i<count;i++) {
		pthread_create(&(result->workers[i]),NULL,tp_workerThread,(void*)result);
	}	
	return result;
}

extern void tp_destroy(tp_threadpool_t* pool) {
	pool->destroying=true;
	for(unsigned int i=0;i<pool->count;i++) {
		pthread_join(pool->workers[i],NULL);
	}
	rb_destroy(pool->jobQueue);
}

extern void tp_enqueue(tp_threadpool_t* pool,tp_worker_t* worker,void* arg) {
	tp_task_t *task=malloc(sizeof(tp_task_t));
	*task=(tp_task_t){worker,arg};
	rb_enqueue(pool->jobQueue,task);
}

extern void tp_wait(tp_threadpool_t* pool) {
	fprintf(stderr,"waiting completion\n");
	while(rb_count(pool->jobQueue)>0) {
		pthread_yield();
	}
	/**
	right here the queue **SHOULD** be empty,
	then all that remain is outstanding running tasks
	**/
	usleep(1000);
	while(pool->active>0) {
		pthread_yield();
	}
}

