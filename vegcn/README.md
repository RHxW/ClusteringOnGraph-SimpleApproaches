#vegcn

This is a simple version of `GCN-V` & `GCN-E`(TODO).

The original version can be found here: 
[learn-to-cluster](https://github.com/yl-1993/learn-to-cluster).



## The one when GCNV doesn't work
I have a powerful face recognition model(Arcface), and I trained the GCNV model on Glint360k dataset(feature extracted by our recognition model).

The testset is also from Glint360k dataset, 1500 identities selected, totally 107k pics.
And I tried 2 different method on inference, I perform face clustering:
1. using the GCNV features
2. directly on the k-NN graph built from the original features(pruning with a threshold and then label propagation)

And the funny thing is, the second way is better. Which means the GCNV method didn't work at all, and it even gets worse.

Here is my test result:

| \ |GCNV(th=0.62)|origin feature(th=0.55)|
|:---:|---:|---:|
|avg_pre|0.998407|**0.998453**|
|avg_rec|0.946603|**0.958298**|
|fscore|0.971815|**0.977964**|

The code is in:
1. `./inference_gcnv.py`
2. `./cluster_straight_from_knns.py`

### a weaker model
I tested on a much weaker recognition model(unconstrained face dataset, private), and the results goes:

| \ |GCNV|origin feature|
|:---:|---:|---:|
|avg_pre|0.993829|**0.994098**|
|avg_rec|**0.344495**|0.285632|
|fscore|**0.511639**|0.443759|

So it looks like that the GCNV does help on a weak model.

### dimension reduction
I also tried dimension reduction on GCNV features and original features.
And the conclusion is:

GCNV features is much better than the original ones. And the lower the dimension goes, the better GCNV features outperforms.