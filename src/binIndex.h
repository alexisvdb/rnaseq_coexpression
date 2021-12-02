#pragma once
#include "binFormat.h"
/**
Index library with an underlying self balanced tree (red-black)
**/

/**
Opaque struct
**/
typedef struct binIndex binIndex;

/**
Load index from an input file

@params : 
inputFilename : input file path

@return :
index loaded 
**/
extern binIndex* binIndexLoad(const char* inputFilename);
extern binIndex* binIndexLoadStream(FILE* input);
/**
Perform a lookup on index 

@param :
index : index to perform a lookup on
key : key to lookup in the index

@return
integer that correspond to the position of the key in the index
**/
extern index_t binIndexLookup(const binIndex* index,const char* key);

/// No assert version
extern index_t binIndexLookup2(const binIndex* index,const char* key);

/**
get count of entries in the index
**/
extern index_t binIndexGetCount(const binIndex* index);

/**
Load index from input file and keep the reverse lookup index in the structure

@params : 
inputFilename : input file path

@return :
index loaded 
**/
extern binIndex* binIndexLoadRev(const char* inputFilename);

/**
Perform a reverse lookup

@params : 
index : index with reverse lookup data
offset : index of the key

@return :
key corresponding to index
**/
extern char* binIndexReverseLookup(const binIndex* index,const index_t offset);

/**
Unload index from memory
**/
extern void binIndexFree(binIndex* index);

/**
Add entry to index
**/
extern void binIndexAddEntry(binIndex* index,const char* key,index_t value);

/**
Create empty index
**/
extern binIndex* createIndex(void);

/**
Fill presence vector
**/
extern int fillPresenceVector(int* vector,const binIndex* lookup,FILE* input);
