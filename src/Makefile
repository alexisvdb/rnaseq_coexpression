BIN := ../bin
ALL :=	$(BIN)/getExpressionCorrelationBin2Bin \
		$(BIN)/queryBin \
		$(BIN)/makeBinTable3 \
		$(BIN)/makeBinTable4 \
		$(BIN)/printTable \
		$(BIN)/printTable2 \
		$(BIN)/walkfile \
		$(BIN)/expression2bin \
		$(BIN)/printTable3 \
		$(BIN)/getTopCorrelatedSymbols \
		$(BIN)/getTopCorrelatedSymbols2 \
		$(BIN)/getTopCorrelatedProbes \
		$(BIN)/readProbes \
		$(BIN)/enrichment \
		$(BIN)/generateGcClassStats \
		$(BIN)/refseq2symbol2 \
		$(BIN)/significantCorrelatedSymbolCount \
		$(BIN)/addDegrees \
		$(BIN)/symbolCountDegreesPvalue \
		$(BIN)/motifCountAddInformations \
		$(BIN)/motifEnrichmentPvalue \
		$(BIN)/symbolGoCount \
		$(BIN)/goEnrichment \
		$(BIN)/remap \
		$(BIN)/probe2symbolScore
		
WARNINGS := -Wall -Wextra -pedantic -Wshadow -Wpointer-arith -Wcast-align \
            -Wwrite-strings -Wmissing-prototypes -Wmissing-declarations \
            -Wredundant-decls -Wnested-externs -Winline -Wno-long-long \
            -Wuninitialized -Wstrict-prototypes -Wconversion

CFLAGS = $(WARNINGS) -std=c99 -D_GNU_SOURCE -D_POSIX_C_SOURCE=200809L -g -O3 -I Rmath -DMATHLIB_STANDALONE 

.PHONY : all clean

all : $(ALL)

clean : 
	rm -f *.o $(ALL)

$(BIN)/getExpressionCorrelationBin2Bin : getCorrelationBin2Bin.o pearsonsCorrelation.o expressionReader.o utils.o binIndex.o rb_tree.a 
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/getExpressionCorrelationBinMT : getCorrelationBinMT.o pearsonsCorrelation.o expressionReader.o ringBuffer.o rb_tree.a 
	cc $(CFLAGS) -o $@ $^ -lm -lpthread -static
	
$(BIN)/queryBin : queryBin.o utils.o binIndex.o binFormat.o rb_tree.a 
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/makeBinTable3 : makeBinTable3.o utils.o binIndex.o binFormat.o rb_tree.a 
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/makeBinTable4 : makeBinTable3.o utils.o binIndex.o binFormat.o rb_tree.a 
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/printTable : printTable.o utils.o binIndex.o binFormat.o rb_tree.a 
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/walkfile : walkfile.o utils.o binIndex.o rb_tree.a 
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/expression2bin : expression2bin.o expressionReader.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/printTable2 : printTable2.o utils.o binIndex.o binFormat.o rb_tree.a 
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/printTable3 : printTable3.o utils/SingleLinkedList.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/getTopCorrelatedSymbols : getTopCorrelatedSymbols.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/getTopCorrelatedSymbols2 : getTopCorrelatedSymbols2.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/getTopCorrelatedProbes :  getTopCorrelatedProbes.o
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/readProbes :  readProbes.o
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/enrichment : enrichment.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/generateGcClassStats : generateGcClassStats.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/refseq2symbol2 : refseq2symbol2.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/significantCorrelatedSymbolCount : significantCorrelatedSymbolCount.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/addDegrees : addDegrees.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/symbolCountDegreesPvalue : symbolCountDegreesPvalue.o utils.o Rmath/libRmath.a 
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/motifCountAddInformations : motifCountAddInformations.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/motifEnrichmentPvalue : motifEnrichmentPvalue.o utils.o Rmath/libRmath.a 
	cc $(CFLAGS) -o $@ $^ -lm -static

pearsonCorrelationBatchCuda.o : pearsonCorrelationBatchCuda.cu
	nvcc -arch compute_20 -c $^ 

$(BIN)/getExpressionCorrelationCuda : getCorrelationCuda.o pearsonCorrelationBatchCuda.o 
	cc $(CFLAGS) -o $@ $^ -L/usr/local/cuda/lib64 -lcudart

$(BIN)/symbolGoCount : symbolGoCount.o fillPresenceVectorMap.o utils.o binIndex.o rb_tree.a
	cc $(CFLAGS) -o $@ $^ -static

$(BIN)/goEnrichment : goEnrichment.o utils.o binIndex.o rb_tree.a Rmath/libRmath.a
	cc $(CFLAGS) -o $@ $^ -lm -static

$(BIN)/probe2symbolScore : probe2symbolScore.o binIndex.o rb_tree.a utils.o
	cc $(CFLAGS) -o $@ $^ -static


$(BIN)/remap : remap.o
	$(CC) $(CFLAGS) -o $@ $^ -static

