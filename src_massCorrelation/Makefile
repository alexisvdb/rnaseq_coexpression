BIN := ../bin_massCorrelation

ALL :=	$(BIN)/massCorrelation \
	$(BIN)/massCorrelationMatrix \
	$(BIN)/massCorrelationThread \
	$(BIN)/massCorrelationNone \
	$(BIN)/compareBin \
	$(BIN)/splitCorrelation \
	$(BIN)/makeDist


ifneq ($(shell which nvcc),)
ALL +=	$(BIN)/massCorrelationGPU \
	$(BIN)/massCorrelationMatrixGpu 
endif

WARNINGS := -Wall -Wextra -pedantic -Wshadow -Wpointer-arith -Wcast-align \
            -Wwrite-strings -Wmissing-prototypes -Wmissing-declarations \
            -Wredundant-decls -Wnested-externs -Winline -Wno-long-long \
            -Wuninitialized -Wstrict-prototypes -Wconversion 

CFLAGS += $(WARNINGS) -std=c99 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -g -O3 
NVCCFLAGS += 
.PHONY : all clean

%.o : %.cu 
	nvcc -arch compute_20 --ptxas-options=-v --compiler-options "-O2" -c $^ -o $@

all : $(ALL)

clean :
	rm $(ALL) *.o

$(BIN)/massCorrelation : massCorrelation.o arrayIO.o correlationCPU.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

$(BIN)/massCorrelationNone : massCorrelation.o arrayIO.o correlationNone.o
	$(CC) $(CFLAGS) -o $@ $^ 

$(BIN)/massCorrelationGPU : massCorrelation.o arrayIO.o correlationGPUi.o
	$(CC) $(CFLAGS) -o $@ $^ -lm -L/usr/local/cuda/lib64 -lcudart

#correlationGPU.o : correlationGPU.cu
#	nvcc -arch compute_20 --ptxas-options=-v --compiler-options "-O2" -c $^ -o $@

$(BIN)/massCorrelationMatrix : massCorrelation.o arrayIO.o correlationMatrixCpu.o
	$(CC) $(CFLAGS) -o $@ $^ -lm

#correlationMatrixGpu.o : correlationMatrixGpu.cu
#	nvcc -arch compute_20 --ptxas-options=-v --compiler-options "-O2" -c $^ -o $@

$(BIN)/massCorrelationMatrixGpu : massCorrelation.o arrayIO.o correlationMatrixGpu.o
	$(CC) $(CFLAGS) -o $@ $^ -lm -L/usr/local/cuda/lib64 -lcudart

$(BIN)/massCorrelationThread : massCorrelation.o arrayIO.o correlationCPUThreads.o threadPool.o ringBuffer.o
	$(CC) $(CFLAGS) -o $@ $^ -lm -lpthread

$(BIN)/compareBin : compareBin.o
	$(CC) $(CFLAGS) -o $@ $^ -lm
	
correlationOMP.o : correlationOMP.c
	$(CC) $(CFLAGS) -c -fopenmp -o $@ $^ 

$(BIN)/massCorrelationOMP : massCorrelation.o arrayIO.o correlationOMP.o
	$(CC) $(CFLAGS) -fopenmp -o $@ $^ -lm

$(BIN)/array2bin : array2bin.o arrayIO.o
	$(CC) $(CFLAGS) -o $@ $^ 

$(BIN)/splitCorrelation : splitCorrelation.o 
	$(CC) $(CFLAGS) -o $@ $^ 

$(BIN)/makeDist :makeDist.o
	$(CC) $(CFLAGS) -o $@ $^ 
