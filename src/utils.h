#pragma once
#include <stdio.h>
#include "binFormat.h"

/*
Callback for readlineLoop
@param : line : string read from input file without the newline character
@param : size : line size
@param : param : custom param
@return int - not used 
*/
typedef int(readlineCallbackProc_t)(char* line,size_t size,void* param);

/*
TrimCR
Remove new line charactera at the end of the string ( replacing it with a '\0' )
@param : subject : char* string to remove the newline character
@param : maxLength : string length
@return : position where the newline character was found and replaced
*/
extern int trimCR(char* subject,const size_t maxLength);

/*
readlineLoop 
read file line by line and line and trim the newline character from the line and call the user function 
@param : input : FILE input to read from
@param : readlineCallbackProc : user function to call with each line
@param : custom parameter
*/
extern int readlineLoop(FILE* input,readlineCallbackProc_t readlineCallbackProc,void* params);
