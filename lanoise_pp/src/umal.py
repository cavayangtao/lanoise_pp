import numpy as np
import matplotlib.pyplot as pl
import torch
import torch.nn as nn
import torch.utils.data
import time
import shutil
import scipy.special
import earlystop
import argparse

class elu_modified(nn.Module):
    def __init__(self, alpha=1.0, shift=5.0, epsilon=1e-7):
        super(elu_modified, self).__init__()
        
        self.alpha = alpha
        self.shift = shift
        self.epsilon = epsilon

        self.elu = nn.ELU(alpha=alpha)

    def forward(self, x):
        return self.elu(x+self.shift) + 1.0 + self.epsilon


class internal_network(nn.Module):
    def __init__(self, n_dim):
        super(internal_network, self).__init__()

        self.n_dim = n_dim

        self.FC1 = nn.Linear(self.n_dim+1, 120)
        self.FC2 = nn.Linear(120, 60)
        self.FC3 = nn.Linear(60, 10)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        out = self.relu(self.FC1(x))
        out = self.relu(self.FC2(out))
        out = self.relu(self.FC3(out))

        return out

class umal_network(nn.Module):
    def __init__(self, internal_network, n_dim):
        super(umal_network, self).__init__()

        self.n_dim = n_dim

        self.internal = internal_network(self.n_dim)
        self.elu_modified = elu_modified(shift=1e-3)
        self.FC1 = nn.Linear(10, 1)
        self.FC2 = nn.Linear(10, 1)

    def forward(self, x):

        out = self.internal(x)

        mu = self.FC1(out)
        b = self.elu_modified(self.FC2(out))

        return mu, b

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename+'.best')

def ald_log_pdf(y, mu, b, tau):
    """
    Logarithm of the Asymmetric Laplace Probability density function
    """
    return np.where(
        y > mu,
        np.log(tau) + np.log(1 - tau) - np.log(b) - tau * (y - mu) / b,
        np.log(tau) + np.log(1 - tau) - np.log(b) - (tau - 1) * (y - mu) / b)

def minmax(pred, desv_from_minmax = 4):
    """
    For visualization part: Normal assumption of min-max values taking the 
     desv_from_minmax*sigmas deviation
    """
    pos_min = np.argmin(pred[0, :, :].flatten() - desv_from_minmax * pred[1, :, :].flatten())
    pos_max = np.argmax(pred[0, :, :].flatten() + desv_from_minmax * pred[1, :, :].flatten())

    return pred[0, :, :].flatten()[pos_min] - desv_from_minmax * pred[1, :, :].flatten()[pos_min], pred[0, :, :].flatten()[pos_max] + desv_from_minmax * pred[1, :, :].flatten()[pos_max]


class umal(object):

    def __init__(self, n_dim=3):
        self.n_dim = n_dim

    def init_training(self, architecture=None, batch_size=100, gpu=0, n_taus=100):
        self.cuda = torch.cuda.is_available()
        self.batch_size = batch_size
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")
                       
        kwargs = {'num_workers': 2, 'pin_memory': True} if self.cuda else {}
        
        # Read synthesis network and fix parameters
        print("Defining NN...")
        self.model = umal_network(architecture, n_dim=self.n_dim).to(self.device)

        self.dataset_train = torch.utils.data.TensorDataset(self.x_train, self.y_train)
        self.dataset_valid = torch.utils.data.TensorDataset(self.x_valid, self.y_valid)

        self.train_loader = torch.utils.data.DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, drop_last=False, **kwargs)
        self.valid_loader = torch.utils.data.DataLoader(self.dataset_valid, batch_size=self.batch_size, shuffle=True, drop_last=False, **kwargs)    

        self.tau_down = 1e-2
        self.tau_up = 1.0 - 1e-2
        self.n_taus = n_taus
        self.n_taus_tensor = torch.as_tensor(np.log(self.n_taus).astype('float32'))

    def init_prediction(self, architecture=None, batch_size=100, gpu=0):
        self.cuda = torch.cuda.is_available()
        self.batch_size = batch_size
        self.gpu = gpu
        self.device = torch.device(f"cuda:{self.gpu}" if self.cuda else "cpu")

        kwargs = {'num_workers': 2, 'pin_memory': True} if self.cuda else {}

        # Read synthesis network and fix parameters
        print("Defining NN...")
        self.model = umal_network(architecture, n_dim=self.n_dim).to(self.device)

    def load_weights(self, checkpoint):
        
        self.checkpoint = checkpoint
        
        tmp = torch.load(self.checkpoint, map_location=lambda storage, loc: storage)          
        
        self.model.load_state_dict(tmp['state_dict'])        
        print("=> loaded checkpoint from '{}'".format(self.checkpoint))
        self.model.eval()

    def optimize(self, epochs, lr=1e-4, smooth=0.05, patience=200):

        self.lr = lr
        self.n_epochs = epochs
        self.smooth = smooth
        self.patience = patience
        
        root = 'weights'

        current_time = time.strftime("%Y-%m-%d-%H:%M:%S")
        self.out_name = '{2}/{0}_-lr_{1}'.format(current_time, self.lr, root)

        print("Network name : {0}".format(self.out_name))
                
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                
        self.loss = []        
        self.loss_val = []
        best_loss = -1e10

        trainF = open('{0}.loss.csv'.format(self.out_name, self.lr), 'w')

        early_stopping = earlystop.EarlyStopping(patience=self.patience, verbose=True)

        for epoch in range(1, epochs + 1):
            
            self.train(epoch)
            self.test()

            print(f'Epoch {epoch} - loss = {self.loss[-1]} - loss_val = {self.loss_val[-1]}')

            trainF.write('{},{},{}\n'.format(
                epoch, self.loss[-1], self.loss_val[-1]))
            trainF.flush()

            is_best = self.loss_val[-1] < best_loss
            best_loss = max(self.loss_val[-1], best_loss)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.state_dict(),
                'best_loss': best_loss,
                'optimizer': self.optimizer.state_dict(),
                'lr': self.lr
            }, is_best, filename='{0}.pth'.format(self.out_name, self.lr))
            
            early_stopping(self.loss_val[-1], self.model)            

            if (early_stopping.early_stop):
                print("Early stopping")
                break
            
        trainF.close()

    def umal_log_pdf(self, y_true, mu, b, tau):
        error = y_true[:,None,:] - mu
        log_like = torch.log(tau) + torch.log(1.0-tau) - torch.log(b) - torch.max(tau * error, (tau-1.0) * error) / b
        sums = torch.logsumexp(log_like, dim=1) - self.n_taus_tensor
        out = torch.sum(sums, dim=1)
        out = -torch.mean(out)
        
        return out

    def train(self, epoch):

        self.model.train()
                
        loss_avg = 0.0
        
        n = 1

        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']

        for batch_idx, (x, y) in enumerate(self.train_loader):

            x, y = x.to(self.device), y.to(self.device)

            batch_size = x.shape[0]

            tau = (self.tau_up - self.tau_down) * torch.rand(batch_size, self.n_taus, 1)
            tau = tau.to(self.device)
            
            x_repeat = x.view(batch_size, 1, self.n_dim).expand(batch_size, self.n_taus, self.n_dim)
            tmp = torch.cat([x_repeat, tau], dim=2)
            tmp = tmp.view(batch_size*self.n_taus, -1)
            
            self.optimizer.zero_grad()
            
            mu, b = self.model(tmp)

            mu = mu.view(batch_size, self.n_taus, 1)
            b = b.view(batch_size, self.n_taus, 1)
                        
            loss = self.umal_log_pdf(y, mu, b, tau)

            loss.backward()

            self.optimizer.step()

            if (batch_idx == 0):
                loss_avg = loss.item()                
            else:
                loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg                
                                                                        
        self.loss.append(loss_avg)

    def test(self):
        self.model.eval()
                
        loss_avg = 0.0
        
        n = 1
        
        for param_group in self.optimizer.param_groups:
            current_lr = param_group['lr']
        
        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.valid_loader):

                x, y = x.to(self.device), y.to(self.device)

                batch_size = x.shape[0]

                tau = (self.tau_up - self.tau_down) * torch.rand(batch_size, self.n_taus, 1)
                tau = tau.to(self.device)
                
                x_repeat = x.view(batch_size, 1, self.n_dim).expand(batch_size, self.n_taus, self.n_dim)
                tmp = torch.cat([x_repeat, tau], dim=2)
                tmp = tmp.view(batch_size*self.n_taus, -1)
                
                self.optimizer.zero_grad()
                
                mu, b = self.model(tmp)

                mu = mu.view(batch_size, self.n_taus, 1)
                b = b.view(batch_size, self.n_taus, 1)
                            
                loss = self.umal_log_pdf(y, mu, b, tau)

                if (batch_idx == 0):
                    loss_avg = loss.item()                    
                else:
                    loss_avg = self.smooth * loss.item() + (1.0 - self.smooth) * loss_avg                    
                
                # t.set_postfix(loss=loss_avg, lr=current_lr)

        self.loss_val.append(loss_avg)


    def predict(self, nx=500, ny=200, ntaus=90):
        
        x_synthetic = np.expand_dims(np.linspace(self.x_train_synthetic.min(), self.x_train_synthetic.max(),nx), 1)
        sel_taus = np.linspace(0.+5e-2,1.-5e-2,ntaus)

        taus = np.tile(sel_taus[None, :, None], (nx, 1, 1))

        with torch.no_grad():

            x_synthetic = torch.from_numpy(x_synthetic.astype('float32')).to(self.device)
            taus = torch.from_numpy(taus.astype('float32')).to(self.device)

            x_repeat = x_synthetic.view(nx, 1, self.n_dim).expand(nx, ntaus, self.n_dim)

            tmp = torch.cat([x_repeat, taus], dim=2)
            tmp = tmp.view(nx*ntaus, -1)

            mu, b = self.model(tmp)

            mu = mu.cpu().numpy().reshape((nx,ntaus,1))
            b = b.cpu().numpy().reshape((nx,ntaus,1))
            taus = taus.cpu().numpy().reshape((nx,ntaus,1))

        im = np.zeros((ny,nx))
        y = np.linspace(-10,10,ny)
        for i in range(ny):
            im[i,:] = scipy.special.logsumexp(ald_log_pdf(y[i], mu[:,:,0], b[:,:,0], taus[:,:,0]), axis=1) - np.log(ntaus)

        im = np.clip(np.flip(np.exp(im),axis=0),0.,0.2)

        f, ax = pl.subplots(figsize=(15,5))

        ax.imshow(im, cmap=pl.cm.Blues, interpolation='none',
           extent=[self.x_train_synthetic.min(), self.x_train_synthetic.max(),-10, 10], aspect="auto")

        sd = ax.scatter(self.x_train_synthetic, self.y_train_synthetic,c='orange', label='synthetic data',alpha=0.6)

        pl.show()

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='Train UMAL')
#     parser.add_argument('--lr', '--learning-rate', default=1e-2, type=float,
#                     metavar='LR', help='Learning rate')
#     parser.add_argument('--epochs', '--epochs', default=2000, type=int,
#                     metavar='Epochs', help='Epochs')
#     parser.add_argument('--batch', '--batch', default=1000, type=int,
#                     metavar='Batch size', help='Batch size')
#     parser.add_argument('--ntaus', '--ntaus', default=100, type=int,
#                     metavar='Epochs', help='Epochs')
#     parser.add_argument('--patience', '--patience', default=200, type=int,
#                     metavar='Patience', help='Patience')
#
#     parsed = vars(parser.parse_args())
#
#     tmp = umal()
#     tmp.init_training(architecture=internal_network, batch_size=parsed['batch'], n_taus=parsed['ntaus'])
#     tmp.optimize(parsed['epochs'], lr=parsed['lr'], patience=parsed['patience'])