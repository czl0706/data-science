import sys
import json
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


class TestDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.trans = transforms.Compose([transforms.ToTensor(),
                                         transforms.Resize((224, 224)),
                                         transforms.Normalize(mean = (0.485, 0.456, 0.406), 
                                                              std = (0.229, 0.224, 0.225))])
    
    def __len__(self):
        return len(self.path)
    
    def __getitem__(self, idx):      
        img = Image.open(self.path[idx])
        try:
            img = img.convert('RGB')
        except Exception as e:
            print(f'excpetion: {e}, path: {self.path[idx]}')
            img = Image.new('RGB', (224, 224))
        img = self.trans(img)
        
        return img

def main():
    # python main.py {test_file}.json
    if len(sys.argv) != 2:
        print("Usage: python main.py {test_file}.json")
        return
    
    from model import model

    # load json 
    with open(sys.argv[1], "r") as f:
        data = json.load(f)
        path = data['image_paths']
        
    # predict
    dataset = TestDataset(path)
    test_dl = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False, num_workers=56)
    
    weight_path = './subdirectory/ckpt.pth'
    # weight_path = './ckpt.pth'
    model.load_state_dict(torch.load(weight_path), strict=False)
    model = model.to('cuda')
    model.eval()
    
    preds = []
    for x in tqdm(test_dl):
        with torch.no_grad():
            # get max value index
            pred = model(x.cuda()).argmax(dim=1).item()
            preds.append(pred)
            
    result = {
        'image_predictions': preds
    }

    with open('image_predictions.json', 'w') as f:
        json.dump(result, f, indent=4)

if __name__ == "__main__":
    main()