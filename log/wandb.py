import os
import torch
import wandb
from wandb import AlertLevel
from .interface import ExperimentLogger

class WandbLogger(ExperimentLogger):
    """Save experiment logs to wandb
    """
    def __init__(self, path, name, group, project, entity, tags, api_key, script=None, save_dir = None):
        os.system(f"wandb login {api_key}")
        self.paths = {
            'model' : f'{save_dir}/model'
        }
        self.run = wandb.init(
                group=group,
                project=project,
                entity=entity,
                tags=tags,
                dir=save_dir
            )
        wandb.run.name = name
        # upload zip file
        wandb.save(save_dir + "/script/script.zip")


    def log_metric(self, name, value, log_step=None, epoch_step=None):
        if epoch_step is not None:
            wandb.log({
                name: value,
                'epoch_step': epoch_step})
        elif log_step is not None:
            wandb.log({
                name: value,
                'log_step': log_step})

    def log_text(self, name, text):
        pass

    def log_image(self, name, image):
        wandb.log({name: [wandb.Image(image)]})

    def log_parameter(self, dictionary):
        wandb.config.update(dictionary)

    def save_model(self, name, state_dict):
        path = f'{self.paths["model"]}/BestModel.pt'
        wandb.save(path)

    def finish(self):
        wandb.finish()