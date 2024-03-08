from torchvision.models import efficientnet_b3
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torch.nn as nn

def get_model():
    def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)

    WeightsEnum.get_state_dict = get_state_dict

    model = efficientnet_b3(weights='DEFAULT')
    model.classifier._modules['1'] = nn.Linear(1536, 2)
    
    return model

model = get_model()