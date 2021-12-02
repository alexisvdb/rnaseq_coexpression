
#include <stdlib.h>
#include "SingleLinkedList.h"

typedef struct singleLinkedList_node node;

struct singleLinkedList 
{
	node* head;
};

struct singleLinkedList_node {
	void* data;
	node *next;	
};

/**
Create a new list head
**/
singleLinkedList* singleLinkedList_init(void) {
	singleLinkedList *result=(singleLinkedList*)malloc(sizeof(singleLinkedList));
	result->head=NULL;
	return result;
}

/**
Push new element on list head

@Params : 
list : singleLinkedList target list
data : data to put at the beginning of the list
**/
void singleLinkedList_push(singleLinkedList* list,void* data) {
	node* insert=(node*)malloc(sizeof(node));
	insert->data=data;
	insert->next=list->head;
	list->head=insert;
}

/**
Pop the top element from the list and return it

@Params :
list : target list

@return :
	Element popped from the list
**/
void* singleLinkedList_pop(singleLinkedList* list) {
	node* target=list->head;
	void* result=target->data;
	list->head=target->next;
	free(target);
	return result;
}

/**
Put a new elemnt at the end of the list 

Has a O(n) complexity 

@Params :
list : target list
data : data to put at the end of the list
**/
void singleLinkedList_append(singleLinkedList* list,void* data) {
	node* target=list->head;
	if(target==NULL) {
		singleLinkedList_push(list,data);
		return;
	}

	node* insert=(node*)malloc(sizeof(node));
	insert->data=data;
	insert->next=NULL;

	while(target->next != NULL) {
		target=target->next;
	}
	target->next=insert;
}

/**
Get the nth element in the list

Has a O(n) complexity

@Params :
list : target list
index : index of the element to get the value of

@return :
	Element value at the specified index
**/
void* singleLinkedList_get(const singleLinkedList* list,const int index) {
	node* target=list->head;
	int i=0;
	while(i<index) {
		if(target==NULL) {return NULL;}
		target=target->next;
		i++;
	}
	if(target==NULL) {return NULL;}
	return target->data;
}

/**
Get count of element in the list

Has a O(n) complexity 

@Params :
list : list to count element of

@return :
	count of element in the list
**/
int singleLinkedList_count(const singleLinkedList* list) {
	int result=0;
	node* target=list->head;
	if(target==NULL) {
		return result;
	}
	//result++;
	while(target!=NULL) {
		result++;
		target=target->next;
	}
	return result;
}
