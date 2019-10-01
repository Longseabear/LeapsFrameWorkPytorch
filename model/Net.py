import torch.nn as nn
from util.ops_normal import *
import os
import shutil
from abc import abstractmethod


class BaseNet(nn.Module):
    def __init__(self, config, name):
        self.name = name

        super(BaseNet, self).__init__()
        self.config = config
        self.param = config.MODEL_PARAM
        self.trainer = None

    def keep_only_maximum_checkpoint(self, checkpoint_root, model_name):
        maximum = self.config.KEEP_LATEST_EPOCH
        filenames = os.listdir(checkpoint_root)
        file_list = []
        for filename in filenames:
            file_path = os.path.join(checkpoint_root, filename)
            if os.path.isdir(file_path) and filename.startswith(model_name):
                file_list.append(filename)
        file_list.sort()

        file_list = file_list[:-maximum]
        for target in file_list:
            shutil.rmtree(os.path.join(checkpoint_root, target))

    def save(self):
        model_name = get_original_model_name(self.config.MODEL_NAME)
        self.config.MODEL_NAME = model_name + "_{:04}".format(self.config['EPOCH'])

        checkpoint_folder = os.path.join(self.config.CHECKPOINT_PATH, self.config.MODEL_NAME)
        checkpoint_filename = "checkpoint.tar"
        config_filename = "net_info.yml"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_filename)
        configuration_path = os.path.join(checkpoint_folder, config_filename)

        if not os.path.isfile(checkpoint_path):
            os.makedirs(checkpoint_folder, exist_ok=True)

        # config file change
        modify_yaml(self.config.CONFIG_PATH, configuration_path, self.config)
        self.config.CONFIG_PATH = configuration_path


        # save checkpoint
        save_item = dict()
        save_item[self.name] = self.state_dict()
        save_item[self.trainer.name] = self.trainer.optimizer.state_dict()

        torch.save(save_item, checkpoint_path)

        # clear checkpoint
        self.keep_only_maximum_checkpoint(self.config.CHECKPOINT_PATH, model_name)

    def load(self):
        checkpoint = None
        checkpoint_folder = os.path.join(self.config.CHECKPOINT_PATH, self.config.MODEL_NAME+'_{0:04d}'.format(self.config.EPOCH))

        checkpoint_filename = "checkpoint.tar"
        checkpoint_path = os.path.join(checkpoint_folder, checkpoint_filename)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.load_state_dict(checkpoint[self.name], strict=True)
            if self.trainer is not None:
                self.trainer.optimizer.load_state_dict(checkpoint[self.trainer.name])
        else:
            print("[INFO] No checkpoint found at '{}'. check if it is first training.".format(checkpoint_path))

    def print_model_param(self):
        for param_tensor in self.state_dict():
            print(param_tensor, self.state_dict()[param_tensor].size())

    @abstractmethod
    def set_trainer(self, data_loader):
        pass # self.trainer = Trainer(self, data_loader)

