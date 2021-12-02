#include <stdio.h>


int main()
{
	const int kb = 1024;
	const int mb = kb * kb;

	//cout << "CUDA version:   v" << CUDART_VERSION << endl;    
	//cout << "Thrust version: v" << THRUST_MAJOR_VERSION << "." << THRUST_MINOR_VERSION << endl << endl; 

	int devCount;
	cudaGetDeviceCount(&devCount);
	//cout << "CUDA Devices: " << endl << endl;
	printf("%i CUDA Devices detected\n\n",devCount);

	for(int i = 0; i < devCount; ++i)
	{
		cudaDeviceProp props;
		cudaGetDeviceProperties(&props, i);
		printf("Device #%i %s : %i.%i :\n"
				"Global memory\t: %i mb\n"
				"Shared memory\t: %i kb\n"
				"Constant memory\t: %i kb\n"
				"Block registers\t: %i\n"
				"Warp size\t: %i\n"
				"Thread per block\t: %i\n"
				"Max block\t: %i,%i,%i\n"
				"Max grid\t: %i,%i,%i\n\n"
				,
				i,props.name,props.major,props.minor,
				props.totalGlobalMem/mb,
				props.sharedMemPerBlock/kb,
				props.totalConstMem/kb,
				props.regsPerBlock,
				props.warpSize,
				props.maxThreadsPerBlock,
				props.maxThreadsDim[0],props.maxThreadsDim[1],props.maxThreadsDim[2],
				props.maxGridSize[0],props.maxGridSize[1],props.maxGridSize[2]
				);
			/*
		cout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
		cout << "  Global memory:   " << props.totalGlobalMem / mb << "mb" << endl;
		cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
		cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
		cout << "  Block registers: " << props.regsPerBlock << endl << endl;

		cout << "  Warp size:         " << props.warpSize << endl;
		cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
		cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1]  << ", " << props.maxThreadsDim[2] << " ]" << endl;
		cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1]  << ", " << props.maxGridSize[2] << " ]" << endl;
		cout << endl;
		*/
	}
}

