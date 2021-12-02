#include <stdlib.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
#include "binIndex.h"
#include "utils.h"
#include "red_black_tree.h"

struct binIndex {
	rb_red_blk_tree* tree;
	index_t count;
	char** revLookup;
};

/**
	Red black tree support function
**/
static void keyDest(void* a) {
	free(a);
}

static int keyComp(const void* a,const void* b) {
	int r=strcmp(a,b);
	if(r==0) {
		return 0;
	}
	return (r>0)?1:-1;
}

static void keyPrint(const void* a) {
	printf("%s",(char*)a);
}

static void valPrint(void* a) {
	printf("%i",*(index_t*)a);
}

static void valDest(void *a){
	free(a);
}

/**
	Helper functions
**/

typedef struct linkedNode {
	struct linkedNode* next;
	char* value;
} linkedNode;

static linkedNode* makeNode(const char* value) {
	linkedNode* result=malloc(sizeof(linkedNode));
	assert(result!=NULL);
	result->value=(char*)value;
	result->next=NULL;
	return result;
}

typedef struct  {
	char** values;
	int count;
} lookupArray;

static lookupArray* loadLookupArray(const char* inputFilename) {
	FILE* input=fopen(inputFilename,"r");
	assert(input!=NULL /*error opening input file*/);
	char* line=NULL; //getline
	size_t size=0;
	linkedNode* root=NULL;
	linkedNode* current=root;
	int count=0;
	while(getline(&line,&size,input)!=-1) {
		//int n=strlen(line);
		//assert(size>2);
		//if(line[size-1]=='\n') {line[size-1]=0;}
		trimCR(line,size);
		if(current==NULL) {
			root=makeNode(strdup(line));
			current=root;
		}else{
			current->next=makeNode(strdup(line));
			current=current->next;
		}
		count++;
		free(line);
		line=NULL;
		size=0;
	}
	fclose(input);
	lookupArray* result=(lookupArray*)malloc(sizeof(lookupArray));
	assert(count!=0);
	result->count=count;
	result->values=malloc(sizeof(char*)*count);
	current=root;
	int index=0;
	while(current!=NULL) {
	//for(int i=0;i<count;i++) {
		result->values[index++]=current->value;
		linkedNode* old=current;
		current=current->next;
		free(old);
	}
	
	return result;
}

static rb_red_blk_tree* loadLookupFromArray(char** input,int count) {
	rb_red_blk_tree* result=RBTreeCreate(keyComp,keyDest,valDest,keyPrint,valPrint);
	assert(result!=NULL /*alloc error*/);
	
	for(int i=0;i<count;i++) {
		index_t* value=malloc(sizeof(index_t));
		*value=i;		
		rb_red_blk_node* r=RBTreeInsert(result,input[i],value);
		assert(r!=NULL /*insert failed*/);
	}
	return result;
}

static index_t loadLookupTreeStream(rb_red_blk_tree* target,FILE* input) {
	char* line=NULL; //getline
	size_t size=0;
	index_t index=0;
	while(getline(&line,&size,input)!=-1) {
		index_t* value=malloc(sizeof(index_t));
		*value=index;
		trimCR(line,size);
		char* key=strdup(line);

		rb_red_blk_node* r=RBTreeInsert(target,key,value);
		assert(r!=NULL /*insert failed*/);
		index++;
		
		free(line);
		line=NULL;
		size=0;
	}
	return index;	
}

static index_t loadLookupTree(rb_red_blk_tree* target,const char* inputFilename) {
	FILE* input=fopen(inputFilename,"r");
	assert(input!=NULL /*error opening input file*/);
	index_t index=loadLookupTreeStream(target,input);
	fclose(input);
	//RBTreePrint(result);
	return index;
}

static index_t Lookup(rb_red_blk_tree* index,const char* key) {
	rb_red_blk_node* r=RBExactQuery(index,(void*)key);
	if(r==NULL) {
		fprintf(stderr, "%s\n", key);
	}
	assert(r!=NULL/* unknown key */);
	
	index_t result=*(index_t*)(r->info);
	return result;
}

static index_t Lookup2(rb_red_blk_tree* index,const char* key) {
	rb_red_blk_node* r=RBExactQuery(index,(void*)key);
	//assert(r!=NULL/* unknown key */);
	index_t result=0xFFFF;
	if(r!=NULL) {
		result=*(index_t*)(r->info);
	}
	return result;
}

static void binIndexInitStream(binIndex* index,FILE* input) {
	index->tree=RBTreeCreate(keyComp,keyDest,valDest,keyPrint,valPrint);
	assert(index->tree!=NULL);
	index->count=loadLookupTreeStream(index->tree,input);
	index->revLookup=NULL;
}

static void binIndexInit(binIndex* index,const char* inputFilename) {
	//index->tree=loadLookup(inputFilename);
	index->tree=RBTreeCreate(keyComp,keyDest,valDest,keyPrint,valPrint);
	assert(index->tree!=NULL);
	index->count=loadLookupTree(index->tree,inputFilename);
	index->revLookup=NULL;
}

/**
	Exported functions
**/
extern binIndex* binIndexLoadStream(FILE* input) {
	binIndex* result=(binIndex*)malloc(sizeof(binIndex));
	binIndexInitStream(result,input);
	return result;
}

extern binIndex* binIndexLoad(const char* inputFilename) {
	binIndex* result=(binIndex*)malloc(sizeof(binIndex));
	binIndexInit(result,inputFilename);
	
	return result;
}

extern binIndex* binIndexLoadRev(const char* inputFilename) {
	binIndex* result=(binIndex*)malloc(sizeof(binIndex));
	lookupArray* array=loadLookupArray(inputFilename);
	result->revLookup=array->values;
	result->count=array->count;
	result->tree=loadLookupFromArray(result->revLookup,result->count);
	free(array);
	return result;
}

extern index_t binIndexLookup(const binIndex* index,const char* key) {
	return Lookup(index->tree,key);
}

extern index_t binIndexLookup2(const binIndex* index,const char* key) {
	return Lookup2(index->tree,key);
}

extern index_t binIndexGetCount(const binIndex* index) {
	return index->count;
}

extern char* binIndexReverseLookup(const binIndex* index,const index_t offset) {
	assert(index->revLookup!=NULL);
    if(index->revLookup==NULL) {return NULL;}
	return index->revLookup[offset];
}

extern void binIndexFree(binIndex* index) {
    if(index->revLookup!=NULL) {
        for(int i=0;i<index->count;i++) {
            free(index->revLookup[i]);
        }
        free(index->revLookup);
    }
    RBTreeDestroy(index->tree);
    free(index);
}


extern void binIndexAddEntry(binIndex* index,const char* key,index_t value) {

	index_t* pValue=malloc(sizeof(index_t));
    assert(pValue != NULL);
    *pValue=value;

	rb_red_blk_node* r=RBTreeInsert(index->tree,strdup(key),pValue);
	assert(r!=NULL /*insert failed*/);

}

extern binIndex* createIndex(void) {
	binIndex* result=(binIndex*)malloc(sizeof(binIndex));
	result->tree=RBTreeCreate(keyComp,keyDest,valDest,keyPrint,valPrint);
	result->count=0;
	result->revLookup;
	return result;
}

/**
fillPresenceVector 
**/
typedef struct fillPresenceVectorReadLineProcParams_t {
	int result;
	binIndex* lookup;
	index_t count;
	int* vector;
}fillPresenceVectorReadLineProcParams_t;

static int fillPresenceVectorReadLineProc(char* line,size_t size,void* params) {
	fillPresenceVectorReadLineProcParams_t* vparams=(fillPresenceVectorReadLineProcParams_t*) params;
	index_t id=binIndexLookup2(vparams->lookup,line);

	if(id<vparams->count) {
		vparams->vector[id]=1;
		vparams->result++;
	}else{
		fprintf(stderr, "no mapping for [%s]\n", line);
	}
}

extern int fillPresenceVector(int* vector,const binIndex* lookup,FILE* input) {

	fillPresenceVectorReadLineProcParams_t params=(fillPresenceVectorReadLineProcParams_t){
		0,
		lookup,
		binIndexGetCount(lookup),
		vector
	};

	int res=readlineLoop(input,fillPresenceVectorReadLineProc,(void*)&params);

	return params.result;
}
