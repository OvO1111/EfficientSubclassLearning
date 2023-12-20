import os
import json, sys
import shutil
from typing import Any
import numpy as np
import pathlib
from os.path import *
from collections import defaultdict

class DatasetParam:
    n_dim = 0
    n_mode = 1
    n_coarse = 0
    n_fine = 0
    total_num = 0
    legend = None
    mapping = None
    dataset_name = None
    
class ExperimentParam:
    patch_size = None
    batch_size = 0
    labeled_batch_size = 0
    max_iter = 0
    exp_name = None
    pseudo_label = False
    mixup_label = False
    separate_norm = False
    priority_cat = False
    base_lr = 0
    labeled_num = 0
    eval_metric = 0
    restore = False
    
class StaticPaths:
    path_to_dataset = None
    path_to_snapshot = None
    path_to_model = None
    path_to_test = None
    path_to_code = None
    
class NetworkParam:
    base_feature_num = 32
    feature_scale = 2
    image_scale = 2
    is_batchnorm = True
    network_name = None


class BaseParser:
    eval_metrics = {'dsc': 0, 'hd95': 1, 'precision': 2, 'recall': 3}
    
    def __init__(self, args):
        self.dataset = DatasetParam()
        self.exp = ExperimentParam()
        self.path = StaticPaths()
        self.network = NetworkParam()
        self.name = None
        
        self.dataset.total_num = args.total_num
        
        self.exp.patch_size = args.patch_size
        self.exp.batch_size = args.bs
        try:
            self.exp.labeled_batch_size = args.labeled_bs
            self.exp.labeled_num = args.labeled_num
        except AttributeError:
            # this is a fully supervised setting
            self.exp.labeled_batch_size = args.bs
            self.exp.labeled_num = args.total_num
        assert self.exp.labeled_num <= self.dataset.total_num, 'labeled num must <= total num'
        assert self.exp.labeled_batch_size <= self.exp.batch_size, 'labeled bs must <= total bs'
        
        self.exp.max_iter = args.iter
        self.exp.exp_name = args.exp_name
        if args.exp_name == '':
            self.exp.exp_name = f"pseudo{args.pseudo}_mixup{args.mixup}_sn{args.sn}_pc{args.pc}"
        self.exp.pseudo_label = args.pseudo
        self.exp.mixup_label = args.mixup
        self.exp.separate_norm = args.sn
        self.exp.priority_cat = args.pc
        self.exp.base_lr = args.lr
        self.exp.eval_metric = self.eval_metrics[args.eval.lower()]
        self.exp.restore = args.restore
        
        self.path.path_to_dataset = args.data_path
        self.path.path_to_snapshot = join(args.model_path, args.exp_name)
        
        self.network.feature_scale = args.feature_scale
        self.network.is_batchnorm = args.is_batchnorm
        self.network.image_scale = args.image_scale
        self.network.base_feature_num = args.base_feature
        
        if not self._checkdir(self.path.path_to_dataset):
            raise RuntimeError(f"Dataset folder {self.path.path_to_dataset} is nonexistent")
        self._maybe_make_necessary_dirs()
        self._load_or_get_necessary_data()
    
    @staticmethod
    def _checkdir(path):
        return exists(path)
    
    def get_dataset(self):
        raise NotImplementedError
    
    def _maybe_make_necessary_dirs(self):
        self.path.path_to_model = join(self.path.path_to_snapshot, 'model')
        self.path.path_to_test = join(self.path.path_to_snapshot, 'test')
        self.path.path_to_code = join(self.path.path_to_snapshot, 'code')
        
        if exists(self.path.path_to_code):
            shutil.rmtree(self.path.path_to_code)
        
        if not self.exp.restore and exists(self.path.path_to_model) and len(os.listdir(self.path.path_to_model)) > 0:
            x = input('press y if u want to delete old model files\n')
            if x.strip().lower() == 'y':
                print('deleting old files')
                shutil.rmtree(self.path.path_to_model)
                # shutil.rmtree(join(self.path.path_to_snapshot))
            else:
                self.path.path_to_model = self.path.path_to_model + '_temp'
                print(f'preserving old model files, current model path is {self.path.path_to_model}')
        
        os.makedirs(self.path.path_to_test, exist_ok=True)
        os.makedirs(self.path.path_to_model, exist_ok=True)
            
        cur_path = abspath('.')
        shutil.copytree(cur_path, self.path.path_to_code, shutil.ignore_patterns('__pycache__', '.git'))
        
    def _load_or_get_necessary_data(self):
        if exists(join(self.path.path_to_dataset, 'mapping.json')):
            with open(join(self.path.path_to_dataset, 'mapping.json'), 'r') as fp:
                self.dataset.mapping = json.load(fp)
        else:
            print(f'not valid mapping file under dir {self.path.path_to_dataset}, using default mapping fine:Any -> coarse:1')
            self.dataset.mapping = {1: list(range(1, self.dataset.n_fine))}
        self.dataset.dataset_name = self.name
    
    def _dump(self):
        x = lambda : defaultdict(x)
        d = x()
        for name, value in self.dataset.__dict__.items():
            d['dataset'][name] = value
        for name, value in self.exp.__dict__.items():
            d['exp'][name] = value
        for name, value in self.path.__dict__.items():
            d['path'][name] = value
        for name, value in self.network.__dict__.items():
            d['network'][name] = value
        with open(join(self.path.path_to_snapshot, 'param.json'), 'w') as fp:
            json.dump(d, fp)
            
    def __repr__(self):
        log = f"\n\n{self.__class__.__name__.replace('Parser', '').upper()} DATASET PARAMETERS\n\n"
        log += '\n'.join([f"{k}: {v}" for k, v in self.dataset.__dict__.items()])
        log += '\n\nEXPERIMENT PARAMETERS\n\n'
        log += '\n'.join([f"{k}: {v}" for k, v in self.exp.__dict__.items()])
        log += '\n\nNETWORK PARAMETERS\n\n'
        log += '\n'.join([f"{k}: {v}" for k, v in self.network.__dict__.items()])
        log += '\n\nSTATIC PATHS\n\n'
        log += '\n'.join([f"{k}: {v}" for k, v in self.path.__dict__.items()])
        log += '\n\n'
        return log


class ACDCParser(BaseParser):
    name = 'ACDC'
    def __init__(self, args):
        super(ACDCParser, self).__init__(args)
        
        self.dataset.n_dim = 2.5
        self.dataset.n_mode = 1
        self.dataset.n_coarse = 2
        self.dataset.n_fine = 4
        self.dataset.total_num = 1312  # total ACDC slices for 140 cases
        self.dataset.legend = ['ENDO-L', 'EPI-L', 'ENDO-R']
        
        self._dump()
        
    def get_dataset(self, *args, **kwargs):
        from dataloaders.acdc import ACDC
        return ACDC(self, *args, **kwargs)
        
        
class BraTS2021Parser(BaseParser):
    name = 'BraTS2021'
    def __init__(self, args):
        super(BraTS2021Parser, self).__init__(args)
        
        self.dataset.n_dim = 3
        self.dataset.n_mode = 4
        self.dataset.n_coarse = 2
        self.dataset.n_fine = 4
        self.dataset.total_num = 876
        self.dataset.legend = ['NTC', 'PET', 'GD-T']
        
        self._dump()
    
    def get_dataset(self, *args, **kwargs):
        from dataloaders.brats2021 import BraTS2021
        return BraTS2021(self, *args, **kwargs)
        

class Refuge2020Parser(BaseParser):
    name = 'REFUGE2020'
    def __init__(self, args):
        super(Refuge2020Parser, self).__init__(args)
        
        self.dataset.n_dim = 2
        self.dataset.n_mode = 3
        self.dataset.n_coarse = 2
        self.dataset.n_fine = 3
        self.dataset.total_num = min(self.dataset.total_num, 252)
        self.dataset.legend = ['optical-disk', 'optical-cup']
        
        self._dump()
    
    def get_dataset(self, *args, **kwargs):
        from dataloaders.refuge2020 import Refuge2020
        return Refuge2020(self, *args, **kwargs)
    
    
class ProstateParser(BaseParser):
    name = 'Prostate'
    def __init__(self, args):
        super(ProstateParser, self).__init__(args)
        
        self.dataset.n_dim = 2.5
        self.dataset.n_mode = 2
        self.dataset.n_coarse = 2
        self.dataset.n_fine = 3
        self.dataset.total_num = min(self.dataset.total_num, 458)
        self.dataset.legend = ['central_gland', 'peripheral_zone']
        
        self._dump()
    
    def get_dataset(self, *args, **kwargs):
        from dataloaders.prostate import Prostate
        return Prostate(self, *args, **kwargs)
        
        
class Parser:
    def __init__(self, args):
        self.parser = None
        
        if 'acdc' in args.data_path.lower():
            self.parser = ACDCParser(args)
        elif 'brats2021' in args.data_path.lower():
            self.parser = BraTS2021Parser(args)
        elif 'refuge2020' in args.data_path.lower():
            self.parser = Refuge2020Parser(args)
        elif 'prostate' in args.data_path.lower():
            self.parser = ProstateParser(args)
        else:
            raise NotImplementedError
        
    def get_param(self):
        return self.parser
            
    def __repr__(self):
        return self.parser.__repr__()


class OmegaParser:
    def init_from_config(self, conf=None):
        self.path_to_snapshot = conf.experiment.path
        self.path_to_dataset = conf.dataset.path
        
        # dataset
        self.dataset_name = conf.dataset.name
        self.n_coarse = conf.dataset.n_coarse
        self.n_fine = conf.dataset.n_fine
        self.n_dim = conf.dataset.dim
        self.n_channels = conf.dataset.n_channel
        self.mapping = conf.dataset.mapping
        self.ds = conf.dataset.n
        self.legend = conf.dataset.legend
        
        # model
        self.model_name = conf.model.name
        self.p = conf.model.scheme.prior_concatenation
        self.s = conf.model.scheme.separate_normalization
        self.d_p = conf.model.scheme.data_augmentation.pseudo_labeling
        self.d_h = conf.model.scheme.data_augmentation.hierarchical_mix
        self.eval_metric = conf.model.eval
        
        # experiment
        self.exp_name = conf.experiment.name
        self.seed = conf.experiment.seed
        self.labeled_bs = conf.experiment.train.labeled_batch_size
        
        self.bs = conf.experiment.train.batch_size
        self.ps = conf.experiment.train.patch_size
        self.lr = conf.experiment.train.lr
        self.itr = conf.experiment.train.iteration
        self.nl = conf.experiment.train.negative_learning
        self.n_labeled = conf.experiment.train.n_labeled
        
        self.restore = conf.experiment.restore
        self.val_step = conf.experiment.validation.step
        self.val_scheme = conf.experiment.validation.ckpt
        
        self._maybe_make_necessary_dirs()
        return self
        
    def _maybe_make_necessary_dirs(self, destory_on_exist=False):
        self.path_to_model = join(self.path_to_snapshot, 'model')
        self.path_to_test = join(self.path_to_snapshot, 'test')
        self.path_to_code = join(self.path_to_snapshot, 'code')
        
        if exists(self.path_to_code):
            shutil.rmtree(self.path_to_code)
        
        if not self.restore and exists(self.path_to_model) and len(os.listdir(self.path_to_model)) > 0:
            x = input('press y if u want to delete old model files\n')
            if x.strip().lower() == 'y':
                print('deleting old files')
                shutil.rmtree(self.path_to_model)
                # shutil.rmtree(join(self.path.path_to_snapshot))
            else:
                self.path_to_model = self.path_to_model + '_temp'
                print(f'preserving old model files, current model path is {self.path_to_model}')
        
        os.makedirs(self.path_to_test, exist_ok=True)
        os.makedirs(self.path_to_model, exist_ok=True)
            
        cur_path = abspath(os.getcwd())
        shutil.copytree(cur_path, self.path_to_code, shutil.ignore_patterns('__pycache__', '.git'))
        
    def get_dataset(self, *args, **kwargs):
        import importlib
        return importlib.import_module(f"dataloaders.{self.dataset_name}").AliasDataset(self, *args, **kwargs)