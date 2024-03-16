import os

import torch

from .interface import ExperimentLogger

class NeptuneLogger(ExperimentLogger):
    """Save experiment logs to neptune
    See here -> https://neptune.ai/
    """
    def __init__(self, user, token, name, project, tags, description=None, backup=None):
        import neptune
        self.run = neptune.init_run(
            name=name,
            api_token=token,
            project=f"{user}/{project}",
            tags=tags,
            description=description,
            source_files=backup,
            capture_stderr=False,
            capture_hardware_metrics=False
        )
        
    def log_metric(self, name, value, log_step=None, epoch_step=None):
        if log_step:
            self.run[name].append(value=value, step=log_step)
        elif epoch_step:
            self.run[name].append(value=value, step=epoch_step)
        else:
            self.run[name].append(value)
        #self.run[name].append(value)

    def log_text(self, name, text):
        if len(text) < 990:
            self.run[name].log(text)
        else:
            strI = text.split('\n')
            for i in range(len(strI)):
                self.run[name].log(strI[i])
        
    def log_image(self, name, image):
        self.run[name].upload(image)

    def log_parameter(self, dictionary):
        for k, v in dictionary.items():
            self.run[f'parameters/{k}'] = v

    def save_model(self, name, state_dict):
        pass

    def finish(self):
        pass