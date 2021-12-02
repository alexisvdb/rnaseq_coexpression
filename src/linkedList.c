#include "linkedList.h"
#include <stdlib.h>
#include <assert.h>

struct linkNode {
	linkNode_t *next;
	void* data;
};

linkNode_t* linkNodeCreate(void* data) {
	linkNode_t* result=(linkNode_t*)malloc(sizeof(linkNode_t));
	linkNodeSetData(result,data);
	linkNodeSetNext(result,NULL);
	return(result);
}

linkNode_t* linkNodeFree(linkNode_t *node) {
	assert(node!=NULL);
	linkNode_t *next=node->next;
	void* data=linkNodeGetData(node);
	if(data!=NULL) {free(data);}
	free(node);
	return next;
}

void* linkNodeGetData(const linkNode_t *node) {
	assert(node!=NULL);
	return(node->data);
}

void linkNodeSetData(linkNode_t* node,void* data) {
	assert(node!=NULL);
	node->data=data;
}

linkNode_t* linkNodeGetNext(const linkNode_t* node) {
	assert(node!=NULL);
	return node->next;
}

void linkNodeSetNext(linkNode_t* node,linkNode_t* next) {
	assert(node!=NULL);
	node->next=next;
}

void linkNodeWalk(linkNode_t* root,linkNodeWalkCallback* callback,void* param) {
	linkNode_t* current=root;
	while(current!=NULL) {
		linkNode_t *next=current->next;
		int result=callback(current,param);
		if(result==0) {break;}
		current=next;
	}

}