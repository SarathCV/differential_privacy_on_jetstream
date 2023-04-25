#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# !pip install opacus


# In[ ]:





# In[1]:


import warnings
warnings.simplefilter("ignore")

MAX_GRAD_NORM = 1.2
EPSILON = 50.0
DELTA = 1e-5
EPOCHS = 20

LR = 1e-3


# In[2]:


BATCH_SIZE = 512
MAX_PHYSICAL_BATCH_SIZE = 128


# In[3]:


import torch
import torchvision
import torchvision.transforms as transforms

# These values, specific to the CIFAR10 dataset, are assumed to be known.
# If necessary, they can be computed with modest privacy budgets.
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD_DEV = (0.2023, 0.1994, 0.2010)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD_DEV),
])


# In[4]:


#getting training and testing sets
from torchvision.datasets import CIFAR10

DATA_ROOT = '../cifar10'

train_dataset = CIFAR10(
    root=DATA_ROOT, train=True, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
)

test_dataset = CIFAR10(
    root=DATA_ROOT, train=False, download=True, transform=transform)

test_loader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# In[5]:


#Extracting ResNet18 Model
from torchvision import models

model = models.resnet18(num_classes=10)


# In[ ]:


from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(model, strict=False)
errors[-5:]


# In[ ]:


model = ModuleValidator.fix(model)
ModuleValidator.validate(model, strict=False)


# In[6]:


#GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)


# In[7]:


import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.RMSprop(model.parameters(), lr=LR)


# In[8]:


def accuracy(preds, labels):
    return (preds == labels).mean()


# In[ ]:


from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()

model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    epochs=EPOCHS,
    target_epsilon=EPSILON,
    target_delta=DELTA,
    max_grad_norm=MAX_GRAD_NORM,
)

print(f"Using sigma={optimizer.noise_multiplier} and C={MAX_GRAD_NORM}")


# In[9]:


import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager


# In[10]:


def test(model, test_loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            top1_acc.append(acc)

    top1_avg = np.mean(top1_acc)

    print(
        f"\tTest set:"
        f"Loss: {np.mean(losses):.6f} "
        f"Acc: {top1_avg * 100:.6f} "
    )
    lines=[str((np.mean(losses))),str((top1_avg * 100))]
    print(lines)
    with open('/tmp/output1.txt', 'w') as f:
        for line in lines:
            f.write(line)
            f.write('\n')
    return np.mean(top1_acc)


# In[11]:


#state dict
PATH = "model_test_state_dict.pt"
model.load_state_dict(torch.load(PATH, map_location=device))
model.eval()


# In[ ]:


top1_acc = test(model, test_loader, device)


# In[12]:


#Entire model
# PATH = "model_test_entire.pt"
# model = torch.load(PATH, map_location=device)
# model.eval()


# In[13]:


# top1_acc = test(model, test_loader, device)


# In[ ]:




