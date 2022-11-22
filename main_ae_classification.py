import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from utils import *
from data_loader import DepthDatasetLoader

num_epochs = 100
batch_size = 512
     
root_path = "/home/sangbeom/resnet/data/depth1116_new/"

depth_dataset = DepthDatasetLoader(root_path=root_path)
train_set, val_set = train_val_split(depth_dataset, 0.1)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 
class Autoencoderv3(nn.Module):
    def __init__(self):
        super(Autoencoderv3,self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(p=0.1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64*24*48, 4),
            )
        self.softmax = nn.Softmax(dim=1)
        self.decoder = nn.Sequential(
            nn.Linear(4, 64*24*48),
            nn.ReLU(True),
            nn.Unflatten(1, (64, 24, 48)),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=0),
            nn.ReLU(True),
            nn.Dropout2d(p=0.1),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2, padding=1)
            )
        
    def forward(self, x):
        out_en = self.encoder(x)
        out = self.softmax(out_en)
        out = self.decoder(out)
        return out, out_en

if __name__=="__main__":
    model = Autoencoderv3().to(device)
    distance   = nn.MSELoss()
    class_loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    mse_multp = 0.5
    cls_multp = 0.5

    model.train()

    for epoch in range(num_epochs):
        total_mseloss = 0.0
        total_clsloss = 0.0
        for ind, data in enumerate(train_loader):
            img, labels = data[0].reshape(-1, 1, 96, 192).to(device), data[1].to(device) 
            output, output_en = model(img)
            loss_mse = distance(output, img)
            loss_cls = class_loss(output_en, labels)
            loss = (mse_multp * loss_mse) + (cls_multp * loss_cls)  # Combine two losses together
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Track this epoch's loss
            total_mseloss += loss_mse.item()
            total_clsloss += loss_cls.item()

        # Check accuracy on test set after each epoch:
        if (epoch+1)%2==0 or (epoch+1)==num_epochs:
            model.eval()   # Turn off dropout in evaluation mode
            acc = 0.0
            total_samples = 0
            with torch.no_grad():
                val_images, labels  = next(iter(val_loader))
                val_images = val_images.reshape(-1, 1, 96, 192)
                val_images = Variable(val_images.float().to(device))
                val_preds, val_en = model(val_images)
                val_loss = F.mse_loss(val_preds, val_images)
                prob = nn.functional.softmax(val_en, dim = 1)
                pred = torch.max(prob, dim=1)[1].detach().cpu().numpy() # Max prob assigned to class 
                acc += (pred == labels.cpu().numpy()).sum()
                total_samples += labels.shape[0]
            print('Epoch {0}, train loss {1}, val loss {2}'.format(epoch+1,
                                                            round(loss.item(), 6),
                                                            round(val_loss.item(), 6)))
            fig, axs = plt.subplots(2,5, figsize=(15,4))
            val_images = val_images.detach().cpu().numpy()
            val_preds = val_preds.detach().cpu().numpy()
            for i in range(5):
                axs[0][i].matshow(np.reshape(val_images[i, :], (96,192)))
                axs[1][i].matshow(np.reshape(val_preds[i, :], (96,192)))
            plt.savefig("data/convnet/ae_class_eval{}.png".format(epoch+1))
            model.train()   # Enables dropout back again
            print('epoch [{}/{}], loss_mse: {:.4f}  loss_cls: {:.4f}  Acc on test: {:.4f}'.format(epoch+1, num_epochs, total_mseloss / len(train_set), total_clsloss / len(train_set), acc / total_samples))