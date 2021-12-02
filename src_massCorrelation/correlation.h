#pragma once
#ifndef _CORRELATION_H_
#define _CORRELATION_H_

#ifdef __cplusplus
extern "C" {
#endif
/**
 * Get correlations between all the vectors
 *
 * Result will be return in a upper triangular matrix,
 * ommiting correlation of each vector with itself
 * so that only half of the memory space ( slightly less ) 
 * is used. and of course half of the calculation
 *
 * @param [in] vectors 
 * input list of vectors to perform calculations on
 *
 * @param [in] cols
 * vectors length
 *
 * @param [in] rows
 * vectors count
 *
 * @return
 * correlation values betwwen every vectors of the list
 */
extern float* getCorrelation(const float* vectors,const size_t cols,const size_t rows);

#ifdef __cplusplus
}
#endif

#endif

