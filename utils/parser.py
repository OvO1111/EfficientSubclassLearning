import os
import json
import shutil
import numpy as np
from os.path import *
from collections import defaultdict

class DatasetParam:
    n_dim = 0
    n_mode = 1
    n_coarse = 0
    n_fine = 0
    total_num = 0
    legend = None
    
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


class BaseParser:
    eval_metrics = {'dsc': 0, 'hd95': 1, 'precision': 2, 'recall': 3}
    
    def __init__(self, args):
        self.dataset = DatasetParam()
        self.exp = ExperimentParam()
        self.path = StaticPaths()
        self.network = NetworkParam()
        self.logger = None
        
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
        self.exp.pseudo_label = args.p
        self.exp.mixup_label = args.m
        self.exp.separate_norm = args.sn
        self.exp.priority_cat = args.pc
        self.exp.base_lr = args.lr
        self.exp.eval_metric = self.eval_metrics[args.eval.lower()]
        self.exp.restore = args.restore
        
        self.path.path_to_snapshot = join(args.model_path, args.exp_name)
        self.path.path_to_dataset = args.data_path
        
        self.network.feature_scale = args.feature_scale
        self.network.is_batchnorm = args.is_batchnorm
        self.network.image_scale = args.image_scale
        self.network.base_feature_num = args.base_feature
        
        if not self._checkdir(self.path.path_to_dataset):
            raise RuntimeError(f"Dataset folder {self.path.path_to_dataset} is nonexistent")
        self._maybe_make_necessary_dirs()
    
    @staticmethod
    def _checkdir(path):
        return exists(path)
    
    @staticmethod
    def get_dataset():
        raise NotImplementedError
    
    def _maybe_make_necessary_dirs(self):
        self.path.path_to_model = join(self.path.path_to_snapshot, 'train')
        self.path.path_to_test = join(self.path.path_to_snapshot, 'test')
        self.path.path_to_code = join(self.path.path_to_snapshot, 'code')
        
        os.makedirs(self.path.path_to_test, exist_ok=True)
        if exists(self.path.path_to_code):
            shutil.rmtree(self.path.path_to_code)
        if not self.exp.restore and exists(self.path.path_to_model) and len(os.listdir(self.path.path_to_model)) > 0:
            x = input('press y if u want to delete old model files\n')
            if x.strip().lower() == 'y':
                print('deleting old files')
                shutil.rmtree(self.path.path_to_model)
            else:
                self.path.path_to_model = self.path.path_to_model + '_temp'
                print(f'preserving old model files, current model path is {self.path.path_to_model}')
        
        os.makedirs(self.path.path_to_model, exist_ok=True)
            
        cur_path = abspath('.')
        shutil.copytree(cur_path, self.path.path_to_code, shutil.ignore_patterns('__pycache__', '.git'))
    
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
    def __init__(self, args):
        super(ACDCParser, self).__init__(args)
        
        self.dataset.n_dim = 2
        self.dataset.n_mode = 1
        self.dataset.n_coarse = 2
        self.dataset.n_fine = 4
        self.dataset.total_num = 1312  # total ACDC slices for 140 cases
        self.dataset.legend = ['ENDO-L', 'EPI-L', 'ENDO-R']
        
        self._dump()
        
    @staticmethod
    def get_dataset(*args, **kwargs):
        from dataloaders.acdc import ACDC
        return ACDC(*args, **kwargs)
        
        
class BraTS2021Parser(BaseParser):
    def __init__(self, args):
        super(BraTS2021Parser, self).__init__(args)
        
        self.dataset.n_dim = 3
        self.dataset.n_mode = 4
        self.dataset.n_coarse = 2
        self.dataset.n_fine = 4
        self.dataset.total_num = 876
        self.dataset.legend = ['NTC', 'PET', 'GD-T']
        
        self._dump()
    
    @staticmethod
    def get_dataset(*args, **kwargs):
        from dataloaders.brats2021 import BraTS2021
        return BraTS2021(*args, **kwargs)
        

class Refuge2020Parser(BaseParser):
    def __init__(self, args):
        super(Refuge2020Parser, self).__init__(args)
        
        self.dataset.n_dim = 2
        self.dataset.n_mode = 3
        self.dataset.n_coarse = 2
        self.dataset.n_fine = 3
        self.dataset.total_num = 400
        self.dataset.legend = ['optical-disk', 'optical-cup']
        
        self._dump()
    
    @staticmethod
    def get_dataset(*args, **kwargs):
        from dataloaders.refuge2020 import Refuge2020
        return Refuge2020(*args, **kwargs)
        
        
class Parser:
    def __init__(self, args):
        self.parser = None
        self.dataset_name = None
        
        if 'acdc' in args.data_path.lower():
            self.dataset_name = 'acdc'
            self.parser = ACDCParser(args)
        elif 'brats2021' in args.data_path.lower():
            self.dataset_name = 'brats2021'
            self.parser = BraTS2021Parser(args)
        elif 'refuge2020' in args.data_path.lower():
            self.dataset_name = 'refuge2020'
            self.parser = Refuge2020Parser(args)
        else:
            raise NotImplementedError
        
    def get_param(self):
        return self.parser
            
    def __repr__(self):
        return self.parser.__repr__()
    