# EfficientSubclassLearning
Sample code for MICCAI2023 Efficient Subclass Learning

run `run.sh` and set the corresponding parameters to train a model and `test.py` to evaluate its validity (you might want to take a look at how arguments are defined in `train_proposed.py` and `test.py`). Currently configured datasets are ACDC and BraTS2021, be sure to include the correct dataset in the `data_path` argument in `train_**.py`, it is needed for the parser to make the right choice for the dataloader.

<pre>
dataset structure for ACDC:
`ACDC`  
  |----`data`
  |       |----`slices` (for all slices used for training)  
  |       |----`patient_***_frame0*.h5` (you should first package each image and its label in a h5 file)  
  |       |----`...`  
  |----`train_slices.list` (for all train slices, note you should include only the names for `.h5` files)  
  |       |---- (`patient_000_frame01_slice_0\npatient_000_frame01_slice_1\n...`)  
  |----`val.list` (for all val instances)  
  |----`test.list` (for all test instances)  
</pre>
  
and for other datasets:  
<pre>
`<dataset_name>`
  |----`data`  
  |       |----`***.h5` (you should first package each image and its label in a h5 file)  
  |       |----`...`   
  |----`train.list`  
  |----`val.list`  
  |----`test.list`  
</pre>
