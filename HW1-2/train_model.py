# %%
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import wandb

# %%
config = {
  "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
  "batch_size": 128,
  "learning_rate": 0.002,
  'epochs': 50,
  'save_path': './'
}

# %%
wandb.init(project = "Shit-CNN", config = config)

# %%
class HDF5Dataset(Dataset):
    def __init__(self, file_path):
        self.file = h5py.File(file_path, 'r')
        self.images = self.file['images']
        self.labels = self.file['labels']
        self.length = len(self.images)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx]) 
        # 0, 1 to [1, 0], [0, 1]
        label = np.eye(2)[self.labels[idx]]
        label = torch.tensor(label, dtype=torch.float32) 
        return image, label

hdf5_dataset = HDF5Dataset('./custom_dataset.hdf5')
train_dl = DataLoader(hdf5_dataset, 
                      batch_size=config['batch_size'], 
                      shuffle=True, 
                      num_workers=56, 
                      pin_memory=True)

# %%
from torchvision.models import efficientnet_b3
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url
import torch.nn as nn

def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)

WeightsEnum.get_state_dict = get_state_dict

model = efficientnet_b3(weights='DEFAULT')
model.classifier._modules['1'] = nn.Linear(1536, 2)

model = model.to(config['device'])

# %%
from thop import profile
from torchsummary import summary

# pseudo image
image = torch.rand(1, 3, 224, 224).cuda()

out = model(image)

# torchsummary report
summary(model, input_size=(3, 224, 224))
print(f'From input shape: {image.shape} to output shape: {out.shape}')

# thop report
macs, parm = profile(model, inputs=(image, ))
print(f'FLOPS: {macs * 2 / 1e9} G, Params: {parm / 1e6} M.')

# %%
# check the number of label 0 and 1 in the dataset
with h5py.File('./custom_dataset.hdf5', 'r') as f:
    class_0 = np.sum(f['labels'][:] == 0)
    class_1 = np.sum(f['labels'][:] == 1)

# %%
max_class = max(class_0, class_1)
print(f'Weight of class 0: {max_class/class_0}, Weight of class 1: {max_class/class_1}')
# use weight to balance the class
loss_func = nn.CrossEntropyLoss(weight=torch.tensor([max_class/class_0, max_class/class_1]).to(config['device']))
optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'])

# %%
def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_seed(14)

# %%
import numpy as np
from tqdm import tqdm

# initialize tracker for minimum validation loss
valid_loss_min = np.Inf # set initial "min" to infinity
train_loss_min = np.Inf
# initialize history for recording what we want to know
history = []
device, n_epochs, save_path = config['device'], config['epochs'], config['save_path']

early_stop_count = 0

if not os.path.isdir(save_path): 
    os.mkdir(save_path)

for epoch in range(n_epochs):
    torch.cuda.empty_cache()
    # monitor training loss, validation loss and learning rate
    train_loss = 0.0
    # valid_loss = 0.0
    lrs    = []
    result = {'train_loss': [], 'val_loss': [], 'lrs': []}

    # prepare model for training
    model.train()

    #######################
    # train the model #
    #######################
    # for item in train_dl:
    for batch_idx, item in enumerate(tqdm(train_dl)):
        data, target = item
        # data, target = data.to(device), target.to(device)
        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()

        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the loss
        loss = loss_func(output, target)

        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()

        # record learning rate
        lrs.append(optimizer.param_groups[0]['lr'])

        # update running training loss
        train_loss += loss.item() * data.size(0)

    ######################
    # validate the model #
    ######################
    # model.eval()
    # with torch.no_grad():
    #     for batch_idx, item in enumerate(tqdm(val_dl)):
    #         data, target = item['image'], item['keypoints']
    #         # data, target = data.to(device), target.to(device)
    #         data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

    #         # compute predicted outputs by passing inputs to the model
    #         output = model(data)
    #         # calculate the loss
    #         loss = loss_func(output,target)

    #         # update running validation loss
    #         valid_loss += loss.item() * data.size(0)

    # print training/validation statistics
    # calculate average loss over an epoch
    train_loss = train_loss/len(train_dl.dataset)
    result['train_loss'] = train_loss
    # valid_loss = valid_loss/len(val_dl.dataset)
    # result['val_loss'] = valid_loss
    leaning_rate = lrs
    result['lrs'] = leaning_rate
    history.append(result)

    # print('Epoch {:2d}: Learning Rate: {:.6f} Training Loss: {:.6f} Validation Loss:{:.6f}'.format(
    #     epoch+1,
    #     leaning_rate[-1],
    #     train_loss,
    #     valid_loss
    #     ))
    
    print('Epoch {:2d}: Learning Rate: {:.6f} Training Loss: {:.6f}'.format(epoch+1,
                                                                            leaning_rate[-1],
                                                                            train_loss))
    wandb.log({"epoch": epoch+1, "training_loss": train_loss})

    # save model if validation loss has decreased
    # if valid_loss <= valid_loss_min:
    #     # print("Validation loss decreased({:.6f}-->{:.6f}). Saving model ..".format(
    #     # valid_loss_min,
    #     # valid_loss
    #     # ))
    #     # torch.save(model.state_dict(), path)
    #     valid_loss_min = valid_loss
    #     print('Saving checkpoint...')
    #     state = {
    #         'state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict(),
    #         'epoch': epoch,
    #         'valid_loss_min': valid_loss_min }
    #     if not os.path.isdir(save_path): os.mkdir(save_path)
    #     torch.save(state, save_path + 'checkpoint.pth')
    
    if train_loss <= train_loss_min:
        train_loss_min = train_loss
        print('Saving checkpoint...')
        
        torch.save(model.state_dict(), save_path + 'ckpt.pth')
        early_stop_count = 0
    else:
        early_stop_count += 1

    if early_stop_count >= 10:
        print('\nModel is not improving, so we halt the training session.')
        break
    
wandb.finish()