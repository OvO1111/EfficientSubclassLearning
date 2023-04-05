# EfficientSubclassLearning
Sample code for MICCAI2023 Efficient Subclass Learning

run `run.sh` and set the corresponding parameters to train a model and `test.py` to evaluate its validity (you might want to take a look at how arguments are defined in `train_proposed.py` and `test.py`). Currently configured datasets are ACDC and BraTS2021, be sure to include the correct dataset in the `data_path` argument in `train_**.py`, it is needed for the parser to make the right choice for the dataloader.

dataset structure for ACDC:  
`ACDC`  
&nbsp;&nbsp;&nbsp;&nbsp;|----`data`  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`slices` (for all slices used for training)   
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`patient_000_frame01_slice_0.h5`  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`patient_000_frame01_slice_1.h5`  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`...`  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`patient_000_frame01.h5` (you should first package each image and its labels in a h5 file)  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`patient_000_frame02.h5`  
&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`...`  
&nbsp;&nbsp;&nbsp;&nbsp;|----`train_slices.list` (for all train slices, note you should include only the names for `.h5` files)  
&nbsp;&nbsp;&nbsp;&nbsp;|----`val.list` (for all val instances)  
&nbsp;&nbsp;&nbsp;&nbsp;|----`test.list` (for all test instances)  
&nbsp;&nbsp;&nbsp;&nbsp;|----`mapping.json` (for multi-foreground cases)

and for other datasets:  
`<dataset_name>`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`data`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`***.h5` (you should first package each image and its labels in a h5 file)  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`...`   
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`train.list`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`val.list`  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;|----`test.list`  

sample `train_slices.list` contents:<br />
`patient_000_frame01_slice_0\n`<br />
`patient_000_frame01_slice_1\n`<br />
`...`

sample `train.list` contents:<br />
`patient_000_frame01\n`<br />
`patient_000_frame02\n`<br />
`...`

sample `mapping.json` contents:<br />
`{1: [1, 2, 3], 2: [4, 5], 3: [6] }`

combine image and its fine label to h5:  
`image, fine_label -> h5['image'], h5['label'], h5['granularity']=1`  
for image that does not have fine labels:  
`image, coarse_label -> h5['image'], h5['label'], h5['granularity']=0`  
