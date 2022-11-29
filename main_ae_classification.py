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

class ResBlock(nn.Module):
    """
    A two-convolutional layer residual block.
    """
    
    def __init__(self, c_in, c_out, k, s=1, p=1, mode='encode'):
        assert mode in ['encode', 'decode'], "Mode must be either 'encode' or 'decode'."
        super(ResBlock, self).__init__()
        if mode == 'encode':
            self.conv1 = nn.Conv2d(c_in, c_out, k, s, p)
            self.conv2 = nn.Conv2d(c_out, c_out, 3, 1, 1)
            self.relu  = nn.ReLU()
        elif mode == 'decode':
            self.conv1 = nn.ConvTranspose2d(c_in, c_out, k, s, p)
            self.conv2 = nn.ConvTranspose2d(c_out, c_out, 3, 1, 1)
            self.relu = nn.ReLU()
        # self.BN = nn.BatchNorm2d(c_out)
        self.resize = s > 1 or (s == 1 and p == 0) or c_out != c_in
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        conv1 = self.conv1(x)
        relu = self.relu(conv1)
        conv2 = self.conv2(relu)
        if self.resize:
            x = self.conv1(x)
        return self.relu(x + conv2)

class Encoder(nn.Module):
    """
    Encoder class, mainly consisting of three residual blocks.
    """
    
    def __init__(self):
        super(Encoder, self).__init__()
        self.init_conv = nn.Conv2d(1, 16, 3, 1, 1) 
        # self.BN = nn.BatchNorm2d(16)
        self.rb1 = ResBlock(16, 64, 3, 2, 1, 'encode')
        self.rb2 = ResBlock(64, 128, 3, 2, 1, 'encode') 
        # self.rb3 = ResBlock(128, 256, 3, 2, 1, 'encode') 
        self.flatten = nn.Flatten()
        self.lin1 = nn.Linear(128*24*48, 512)
        self.lin2 = nn.Linear(512, 32)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        init_conv = self.relu(self.init_conv(inputs))
        out = self.rb1(init_conv)
        out = self.rb2(out)
        # out = self.rb3(out)
        out = self.flatten(out)
        out = self.relu(self.lin1(out))
        out = self.relu(self.lin2(out))
        return out

class Decoder(nn.Module):
    """
    Decoder class, mainly consisting of two residual blocks.
    """
    
    def __init__(self):
        super(Decoder, self).__init__()
        # self.rb1 = ResBlock(256, 128, 3, 2, 0, 'decode')
        self.rb1 = ResBlock(128, 64, 3, 2, 0, 'decode')
        self.rb2 = ResBlock(64, 16, 3, 2, 1, 'decode') # 16 32 32
        self.de_lin1 = nn.Linear(512, 128*24*48)
        self.de_lin2 = nn.Linear(32, 512)
        self.out_conv = nn.ConvTranspose2d(16, 1, 2, 1, 1) 
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

    def forward(self, inputs):
        out = self.relu(self.de_lin2(inputs))
        out = self.relu(self.de_lin1(out))
        out = out.view(-1, 128, 24, 48)
        out = self.rb1(out)
        out = self.rb2(out)
        # out = self.rb3(out)
        out = self.out_conv(out)
        out = self.tanh(out)
        return out

class Autoencoder(nn.Module):
    """
    Autoencoder class, combines encoder and decoder model.
    """
    
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.label_num = 4
        self.classifier = nn.Sequential(
            nn.Linear(32, self.label_num),
            nn.Softmax(dim=1)
        )
    @property
    def num_params(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        num_p = sum([np.prod(p.size()) for p in model_parameters])
        return num_p
    
    def forward(self, inputs):
        encoded = self.encoder(inputs)
        pred_label = self.classifier(encoded)
        decoded = self.decoder(encoded)
        return decoded, pred_label



if __name__=="__main__":
    num_epochs = 100
    batch_size = 512
    runname = "resnet1121_class"
    root_path = "/home/sangbeom/resnet/data/depth1116_new/"

    depth_dataset = DepthDatasetLoader(root_path=root_path)
    train_set, val_set = train_val_split(depth_dataset, 0.1)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder().to(device)
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
                val_loss = distance(val_preds, val_images)
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
            plt.savefig("data/convnet/ae_class_v2_eval{}.png".format(epoch+1))
            print('epoch [{}/{}], loss_mse: {:.4f}  loss_cls: {:.4f}  Acc on test: {:.4f}'.format(epoch+1, num_epochs, total_mseloss / len(train_set), total_clsloss / len(train_set), acc / total_samples))
            torch.save(model.encoder.state_dict(), 'weights/{}/resnet_encoder{}steps.pth'.format(runname,epoch+1))
            torch.save(model.decoder.state_dict(), 'weights/{}/resnet_decoder{}steps.pth'.format(runname,epoch+1))
            model.train()   # Enables dropout back again