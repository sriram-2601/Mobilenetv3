import torch.nn as nn
from torchvision import models

def get_model(num_classes, model_name='mobilenet_v3_small', pretrained=True):
   
    if model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights='DEFAULT' if pretrained else None)
    elif model_name == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights='DEFAULT' if pretrained else None)
    else:
        raise ValueError("Invalid model name. Choose 'mobilenet_v3_small' or 'mobilenet_v3_large'.")

    # Only replace the classifier if we aren't using the default 1000 ImageNet classes
    if num_classes != 1000:
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
    
    return model
