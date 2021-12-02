#pragma once

typedef struct singleLinkedList singleLinkedList;

singleLinkedList* singleLinkedList_init(void);
void singleLinkedList_push(singleLinkedList* list,void* data);
void* singleLinkedList_pop(singleLinkedList* list);
void singleLinkedList_append(singleLinkedList* list,void* data);

void* singleLinkedList_get(const singleLinkedList* list,const int index);
int singleLinkedList_count(const singleLinkedList* list);
