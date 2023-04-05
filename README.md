# EfficientSubclassLearning
Sample code for MICCAI2023 Efficient Subclass Learning

run `train_plain_unet.py` or `train_proposed.py` and set the corresponding parameters to train a model and `test.py` to evaluate its validity, currently configured datasets are ACDC and BraTS2021, be sure to include the correct dataset in the `data_path` argument in `train.py`, it is needed for the parser to make the right choice for the dataloader.
