#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <string.h>

typedef struct linkedListNodeStruct linkedListNode;

struct linkedListNodeStruct {
	char* item;
	linkedListNode* next; 
};

typedef struct mapEntryStruct {
	char* key;
	linkedListNode* values;
} mapEntry;

const int MAX_LINE=128*1024;

static void mapEntryAddValue(mapEntry* entry,const char* value) {
	if(value==NULL) {return;}
	linkedListNode* node=malloc(sizeof(linkedListNode));
	assert(node!=NULL);
	node->next=entry->values;
	node->item=value;
	entry->values=node;
}

static mapEntry* mapEntryCreate(const char* key,const char* value) {
	mapEntry* result=malloc(sizeof(mapEntry));
	assert(result!=NULL);
	result->key=key;
	result->values=NULL;
	mapEntryAddValue(result,value);
	return result;
}

static int compareMapEntry(const void* a,const void* b) {
	const mapEntry* va=a;
	const mapEntry* vb=b;
	return strcmp(va->key,vb->key);
}

static mapEntry* readMap(const char* path,int* size) {
	int capacity=16;
	mapEntry* result=malloc(sizeof(mapEntry)*capacity);
	assert(result!=NULL);
	int read=0;
	FILE* input = fopen(path,"r");
	assert(input!=NULL);

	char line[MAX_LINE];

	while(fgets(line,MAX_LINE,input)!=NULL) {
		int split=-1;
		int length=-1;
		for(int i=0;i<MAX_LINE;i++) {
			if(line[i]=='\t') {
				if(split==-1) {
					split=i+1;
					line[i]=0;
					continue;
				}
			}
			if(line[i]=='\n') {
				length=i+1;
				line[i]=0;
				break;
			}
		}
		if(split!=-1) {
			char* key=strdup(line);
			char* value=strdup(line+split);
			
			if((read==0)||(strcmp(result[read-1].key,key)!=0)) {
				mapEntry* entry=mapEntryCreate(key,value);
				result[read++]=*entry;
				free(entry);
				//qsort(result,read,sizeof(mapEntry),compareMapEntry);
				if(read==capacity) {
					capacity += 16;
					result=realloc(result,sizeof(mapEntry)*capacity);
					assert(result!=NULL);
				}
			}else{
				mapEntryAddValue(&result[read-1],value);
			}
		}
	}
	fclose(input);
	fprintf(stderr,"%i entry read\n",read);
	assert(read!=0);
	result=realloc(result,sizeof(mapEntry)*read); // trim results
	assert(result!=NULL);
	*size=read;
	return result;
}

static inline void printResult(const char* key,const char* value) {
	if(value!=NULL) {
		fprintf(stdout,"%s\t%s\n",key,value);
	}else{
		fprintf(stdout,"%s\n",key);
	}
}

int main(int argc,char* argv[]) {
	int mapEntryCount=0;
	char* missingPolicy=getenv("missing");
	int missingPolicyPrint=0;
	if(missingPolicy!=NULL) {
		missingPolicyPrint=(strcmp(missingPolicy,"print")==0);
		fprintf(stderr,"[%s]%i\n",missingPolicy,missingPolicyPrint);
	}
	mapEntry* map=readMap(argv[1],&mapEntryCount);
	int missing=0;
	int read=0;
	char line[MAX_LINE];
	while(fgets(line,MAX_LINE,stdin)!=NULL) {
		int split=-1;
		int length=-1;
		for(int i=0;i<MAX_LINE;i++) {
			if(line[i]=='\t') {
				if(split==-1) {
					line[i]=0;
					split=i+1;
					continue;
				}
			}
			if(line[i]=='\n') {
				length=i+1;
				line[i]=0;
				break;
			}
		}
		if(length!=-1) {
			char* key=strdup(line);
			char* value=NULL;
			if(split!=-1) {value=strdup(line+split);}
			mapEntry* search=mapEntryCreate(key,NULL);
			mapEntry* entry=bsearch(search,map,mapEntryCount,sizeof(mapEntry),compareMapEntry);
			if(entry!=NULL) {
				linkedListNode* node=entry->values;
				assert(node!=NULL);
				while(node!=NULL) {
					printResult(node->item,value);
					node=node->next;
				}
			}else{
				if(missingPolicyPrint==1) {
					printResult(key,value);
				}
				missing++;
			}
			free(search);
			free(key);
			free(value);
		}
		read++;
	}
	fprintf(stderr,"%i/%i missing\n",missing,read);
}

