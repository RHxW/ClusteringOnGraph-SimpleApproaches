# ClusteringOnGraph-SimpleApproaches

COGSAs:
**simplified** version of face clustering algorithms with GCNs.

Simple is GOOD!

## What's the difference?
* less encapsulation, higher readability and easier to modify your own code

## What's new?
* add evaluate method `purity&diverse` and `V-measure`.
* k-NN building methods optimized
* add inference using GCNV features and BATCH inference.
* If you have a powerful recognition model, try to cluster straight from k-NN(built from origin features), it is very likely 
to outperform the GCNV method. More detail ---> `./vegcn/README.md` :)

## Requirements
* Python >= 3.6
* Pytorch >= 0.40
* faiss(cpu/gpu)/nmslib
* sklearn
