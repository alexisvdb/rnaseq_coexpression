#pragma once
#include <stdbool.h>

typedef struct rb_ringBuffer rb_ringBuffer_t;

rb_ringBuffer_t* rb_init(size_t capacity);
void rb_destroy(rb_ringBuffer_t* buffer);
void rb_enqueue(rb_ringBuffer_t* buffer,void* item);
void* rb_dequeue(rb_ringBuffer_t* buffer);
size_t rb_count(rb_ringBuffer_t* buffer);
bool rb_isFull(rb_ringBuffer_t* buffer);

bool rb_enqueueNonBlocking(rb_ringBuffer_t* buffer,void* item);
bool rb_dequeueNonBlocking(rb_ringBuffer_t* buffer,void** output);
