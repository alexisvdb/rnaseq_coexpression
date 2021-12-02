#pragma once


typedef void (tp_worker_t )(void*);
typedef struct tp_threadpool tp_threadpool_t;

tp_threadpool_t* tp_init(unsigned int count);
void tp_destroy(tp_threadpool_t* pool);
void tp_enqueue(tp_threadpool_t* pool,tp_worker_t* worker,void* arg);
void tp_wait(tp_threadpool_t* pool);
