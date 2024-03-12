// Create a KVCacheManager object with the specified parameters
KVCacheManager kvCacheManager(numLayers, numHeads, numKvHeads, hiddenSize, tokensPerBlock, maxNumBlocks,
                             maxBatchSize, maxBeamWidth, maxBlocksPerSeq, dtype, stream);

// Allocate a cache block for a sequence
kvCacheManager.allocateBlock(sequence);

// Free a cache block for a sequence
kvCacheManager.freeBlock(sequence);

// Add a token to a sequence
kvCacheManager.addToken(batchSlotIdx);

// Add a sequence to the batch
kvCacheManager.addSequence(batchSlotIdx, inputLength, beamWidth);

// Remove a sequence from the batch
kvCacheManager.removeSequence(batchSlotIdx);

// Get the block pointers for a batch
kvCacheManager.getBlockPointersOfBatch(dstPointers, batchSize, beamWidth);

// Copy the block pointers for a sequence to external memory
kvCacheManager.copyBlockPointers(dstPointers, dstSlotOffset, batchSlotIdx, beamWidth);

// Calculate the number of cache blocks needed to advance a request by one or two iterations
SizeType numBlocks = kvCacheManager.getNeededBlocksOneStep(req, twoStepsLookAhead);

// Calculate the number of cache blocks needed to advance a request to completion
numBlocks = kvCacheManager.getNeededBlocksToCompletion(req);
