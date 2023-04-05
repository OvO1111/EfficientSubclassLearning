# EfficientSubclassLearning
Sample code for MICCAI2023 Efficient Subclass Learning

run `run.sh` and set the corresponding parameters to train a model and `test.py` to evaluate its validity (you might want to take a look at how arguments are defined in `train_proposed.py`, currently configured datasets are ACDC and BraTS2021, be sure to include the correct dataset in the `data_path` argument in `train_**.py`, it is needed for the parser to make the right choice for the dataloader.
