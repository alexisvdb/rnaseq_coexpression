#pragma once
#include <stdio.h>

typedef struct {
	char* key;
	double* samples;
	int nkey;
} expressionRecord;

expressionRecord* readExpressionRecord(char* line,const int cols);
void freeExpressionRecord(expressionRecord* item);
expressionRecord* expressionRecordDup(const expressionRecord* item,const int cols);

expressionRecord* expressionRecordCreate(const int cols);
void readExpressionRecord2(expressionRecord *target,char* line,const int cols);

void readBinaryExpression(expressionRecord *target,const int cols,FILE* input);
int readBinaryExpressionNoAbort(expressionRecord *target,const int cols,FILE* input);
void writeBinaryExpression(expressionRecord *source,const int cols,FILE* output);
expressionRecord* readBinaryExpressionNew(const int cols,FILE* input);

