#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>

typedef struct linkedNode{
	char* value;
	struct linkedNode *next;
} linkedNode;

linkedNode* makeNode(const char* input) {
	linkedNode* result=malloc(sizeof(linkedNode));
	assert(result!=NULL);
	result->value=strdup(input);
	result->next=NULL;
	return(result);
}

int main(int argc, char *argv[]) {
	linkedNode *head=NULL;
	linkedNode *current;
	char* line=NULL; //getline
	size_t size=0;
	int count=0;
	while(getline(&line,&size,stdin)!=-1) {
		count++;
		assert(line!=NULL);
		if(head==NULL) {
			head=makeNode(line);
			current=head;
		}else{
			linkedNode *next=makeNode(line);
			current->next=next;
			current=next;
		}
		free(line);
		line=NULL;
		size=0;
	}
	//printf("%i",count);
	int counter=0;
	current=head;
	while(current!=NULL) {
		linkedNode *child=current->next;
		if(child!=NULL) {
			printf("%s",current->value);
			while(child!=NULL) {
				printf("%s",child->value);
				child=child->next;
			}
		}
		counter++;
		printf("\n");
		current=current->next;
		
	}
	
	//free data
	current=head;
	while(current!=NULL) {
		linkedNode *next=current->next;
		free(current->value);
		free(current);
		current=next;
	}
	return(EXIT_SUCCESS);
}
