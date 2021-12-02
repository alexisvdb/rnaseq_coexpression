/**
@file

Single link list
**/


/// Opaque type
typedef struct linkNode linkNode_t;

/// Link Node Walk CallBack
/**
	@param	data, link node data
	@param	param, linkNodeWalk param

	@return return 1 or 0 to respectively continue or halt the loop
**/
typedef int (linkNodeWalkCallback)(void*,void*);

/**
	Create a link node 

	Allocate and set first data on a link node, 
	deallocation is left to the caller

	@param initial link node data
**/
linkNode_t* linkNodeCreate(void* data);

/**
	free link node allocated memory (data included when not NULL)

	@param link node to deallocate

	@return return next link node pointer
**/
linkNode_t* linkNodeFree(linkNode_t *node);

/**
	Get link node data

	@param link node to read data from

	@return return link node data
**/
void* linkNodeGetData(const linkNode_t *node);

/**
Attach data to link node

@param node to which the data would be attached
@param data to attach
**/
void linkNodeSetData(linkNode_t* node,void* data);

/**
Get next node linked to the attached node

@param node to read the next from

@return next node
**/
linkNode_t* linkNodeGetNext(const linkNode_t* node);

/**

**/
void linkNodeSetNext(linkNode_t* node,linkNode_t* next);


void linkNodeWalk(linkNode_t* root,linkNodeWalkCallback* callback,void* param);
