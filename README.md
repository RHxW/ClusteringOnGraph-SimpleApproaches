# ClusteringOnGraph-SimpleApproaches

COGSAs:
**simplified** version of face clustering algorithms with GCNs.

This repo is based on [learn-to-cluster](https://github.com/yl-1993/learn-to-cluster).

Simple is GOOD!

## What's the difference?
* less encapsulation, higher readability and easier to modify your own code

## What's new?
* add evaluate method `purity&diverse` and `V-measure`.
* k-NN building methods optimized
* add inference using GCNV features and BATCH inference.
* If you have a powerful recognition model, try to cluster straight from k-NN(built from origin features), it might outperform the GCNV method. More detail ---> `./vegcn/README.md` :)

## How to use?
For face enrollment, try the code in `./face_enroll/`

## Requirements
* Python >= 3.6
* Pytorch >= 0.40
* faiss(cpu/gpu)/nmslib
* sklearn
* [FaceDRTools](https://github.com/RHxW/FaceDRTools)