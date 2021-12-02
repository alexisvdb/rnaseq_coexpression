#pragma once

#ifdef __cplusplus
extern "C" {
#endif
	typedef int probeBatchCallBack_t(int probeIndex,float* input,int count,void* extra);
	extern int batchPearsonCorrelationAll(double* input,int cols,int rows,probeBatchCallBack_t callback,void* extra);
#ifdef __cplusplus
}
#endif

