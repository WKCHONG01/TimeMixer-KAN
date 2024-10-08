import yfinance as yf
import torch
import torch.nn as nn 
from torchvision.transforms import Compose, Normalize, ToTensor
from torch.utils.data import DataLoader, TensorDataset

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import logging as logger
import pandas as pd
from config import Model2Config
from model2 import Model # TimeMixer
from config import Model3Config
from  model3 import HybridModel


from data_provider import Data

import time
import os
import numpy as np

import shutil
import logging 

class EarlyStopping:
    def __init__(self, patience =7):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.val_loss_min = 0
        self.early_stop = False
    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_loss is None:
            self.val_loss_min = val_loss
            self.best_loss = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_loss:
            self.counter +=1
            print(f'Early Stopping counter {self.counter} out of {self.patience} ')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0
    def save_checkpoint(self, val_loss, model, path):
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

def get_loader(dataset, df):
    transform = Compose([
        ToTensor(),
        # Normalize(mean = df.mean(), std = df.std())
    ])
    return transform(dataset)


def train_target_split(x, days, ratio, pred_len):
    seq_len, embed_dim = x.shape[0], x.shape[1]
    if(ratio < 1):

        xtrain = x[:int(len(x)*ratio)]
        xtest = x[int(len(x)*ratio):]
        d = days
        
        
        batch_size = len(xtrain)-days-pred_len
        xtrain_arr = torch.empty((batch_size,days,embed_dim))
        ytrain_arr = torch.empty((batch_size,pred_len,embed_dim))
        
        for i in range(0,batch_size):
            train = xtrain[i:d] # timestamp
            target = xtrain[d:d+pred_len] # timestamp
            ytrain_arr[i] = target
            xtrain_arr[i] = train
            d += 1
        d = days
        batch_size = len(xtest)-days-pred_len
        xtest_arr = torch.empty((batch_size,days,embed_dim))
        ytest_arr = torch.empty((batch_size,pred_len,embed_dim))
        
        for i in range(0,batch_size):
            train = xtest[i:d]
            target = xtest[d:d+pred_len]
            ytest_arr[i] = target
            xtest_arr[i] = train
            d += 1
        
        return xtrain_arr.reshape(-1,days,embed_dim), ytrain_arr.reshape(-1,pred_len,embed_dim), xtest_arr.reshape(-1,days,embed_dim), ytest_arr.reshape(-1,pred_len,embed_dim)
    elif (ratio == 1):
        xtest = x[:]
        
        d = days
        batch_size = len(xtest)-days-pred_len
        xtest_arr = torch.empty((batch_size,days,embed_dim))
        ytest_arr = torch.empty((batch_size,pred_len,embed_dim))
        
        for i in range(0,batch_size):
            train = xtest[i:d]
            target = xtest[d:d+pred_len]
            ytest_arr[i] = target
            xtest_arr[i] = train
            d += 1
        
        return None, None, xtest_arr.reshape(-1,days,embed_dim), ytest_arr.reshape(-1,pred_len,embed_dim)
    


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM,self).__init__()
        self.lstm1 = nn.LSTM(input_size = 1, hidden_size= 2, num_layers= 1, bias = False, batch_first=True)
        self.l1 = nn.Linear(50, 1)
    def forward(self, x):
        h0 = torch.zeros(1,x.size(0),2)
        c0 = torch.zeros(1,x.size(0),2)
        x = self.lstm1(x,(h0,c0))
        return x

        
def visualize_data(data, x_label, y_label):
    fig = plt.figure(figsize=(5,5))
    
    # ax = fig.add_subplot(1,2,1)
    plt.plot(data)
    # plt.set_title("Original data")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    return fig 


def adjust_learning_rate(optimizer, scheduler, epoch):
    lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.warning(f'New learning rate: {lr}')
    return optimizer
    

def train(args, train_dataloader, test_dataloader, model, train_epochs, early_stopping, path, writer):
    criterion = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    train_steps = len(train_dataloader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                    max_lr=0.0001, 
                                                    epochs=train_epochs, 
                                                    steps_per_epoch=train_steps)
    time_now = time.time()
    
    training_loss = []
    validation_loss = []
    earlystopping= EarlyStopping(patience=early_stopping)
    model.train()
    for epoch in range(train_epochs):
        iter_count = 0 
        train_loss = []
        epoch_time = time.time()
        for i ,data in enumerate(train_dataloader):
            iter_count += 1    
            optimizer.zero_grad()
            
            in_x, ground = torch.FloatTensor(data[0]).cuda(), torch.FloatTensor(data[1]).cuda()
            if args.m3:
                pred = []
                # pred, total_entropy = model(in_x)
                
                for a in range(in_x.shape[0]):
                    pred.append(model(in_x[a].unsqueeze(0)))
                pred = torch.cat(pred, dim=0)
                
            else:
                pred = model(in_x)
            # logging.warning('input data: {}'.format(in_x))
            
            
           
            # logging.warning('prediction: {}'.format(pred))
            # logging.warning('ground truth: {}'.format(ground))
            loss = criterion(pred, ground)
            # total_loss = loss + 0.00001*total_entropy
            train_loss.append(loss.item())
            # logging.warning('loss: {}'.format(loss.item()))
            
            

            if (i + 1) % 100 == 0:
                logging.warning("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((train_epochs - epoch) * train_steps - i)
                logging.warning('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            
            loss.backward()
            optimizer.step()
            
            
        
        logging.warning("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(train_loss)
        writer.add_scalar('Training Loss', train_loss.item(), epoch + 1)
        training_loss.append([train_loss])
        valid_loss = validate(args, test_dataloader=test_dataloader, model=model, criterion=criterion)
        writer.add_scalar('Validation Loss', valid_loss.item(), epoch + 1)
        validation_loss.append([valid_loss])
        logging.warning("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss))
        earlystopping(val_loss=valid_loss, model=model, path= path)
        if earlystopping.early_stop:
            logging.warning("Early Stopping")
            break
        optimzer = adjust_learning_rate(optimizer=optimizer, scheduler= scheduler, epoch = epoch+1 )
        scheduler.step()
            

        
        

    best_model_path = path + '/' + 'checkpoint.pth'
    model.load_state_dict(torch.load(best_model_path))

    return model, training_loss, validation_loss



def validate(args, test_dataloader, model, criterion):
    valid_loss = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            in_x, ground = torch.FloatTensor(data[0]).cuda(), torch.FloatTensor(data[1]).cuda()
            if args.m3:

                pred = []
                # pred, total_entropy = model(in_x)
                
                for a in range(in_x.shape[0]):
                    pred.append(model(in_x[a].unsqueeze(0)))
                pred = torch.cat(pred, dim=0)
            else:
                pred = model(in_x)
            pred = pred.detach().cpu()
            ground = ground.detach().cpu()
            loss = criterion(pred,ground)
            valid_loss.append(loss.item())


    valid_loss = np.average(valid_loss)
    model.train()
    return valid_loss



def main(args):
    if args.m3:
        
        log_dir = os.path.join(args.save_path, 'model3_embedding2')
        # log_dir = 'D:/ntu/stocks/model3_embedding2'
    elif args.timemixer:
        log_dir = os.path.join(args.save_path, 'model2')
        # log_dir = 'D:/ntu/stocks/model2'
        

    

    # data = yf.Ticker("AU8U.SI")
    
    data = Data("AU8U.SI")
    # Get historical closing prices
    Close = data.extract_data(['Close','Open','High','Low'])
    
    

    '''
    Date = hist.index.date.reshape(-1,1)
    # Date = [[i[0].year,i[0].day,i[0].month] for i in Date]
    Open = hist[['Open']]
    High = hist[['High']]
    Low = hist[['Low']]
    Close = hist[['Close']]
    Volume = hist[['Volume']]
    '''


    c = Close.to_numpy() # timestamp, 1
    c = get_loader(c, Close).squeeze(0) # timestamp, 1 after squeezed before is 1, timestamp, 1
    
    
    model1 = LSTM()

    early_stopping=7
    if args.timemixer:
        path = os.path.join(args.save_path, 'model2_checkpoint')
        # path = "D:/ntu/stocks/model2_checkpoint"

    elif args.m3:
        path = os.path.join(args.save_path, 'model3_embedding2_checkpoint')
        # path = "D:/ntu/stocks/model3_embedding2_checkpoint"
        
    if not os.path.exists(path):
        os.makedirs(path)
    
    if args.timemixer:
        config = Model2Config()
        model = Model(configs=config).cuda()
    # print("Time Mixer")
    # print(model2)
    elif args.m3:
        config = Model3Config()
        model = HybridModel(configs=config).cuda()
        # model.apply(init_weights)
    # print("Model 3")
    # print(model3)
    # exit()
      
    torch.cuda.empty_cache()
    if args.train:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)

        train_epochs = 30
        ctrain_x, ctrain_y, ctest_x, ctest_y  = train_target_split(c,days=config.seq_len, ratio=config.ratio, pred_len=config.pred_len) 
        train_data = TensorDataset(ctrain_x, ctrain_y) 
        test_data = TensorDataset(ctest_x,ctest_y)
        train_dataloader = DataLoader(train_data,batch_size=config.training_batchsize, shuffle=False) #batch, batchsize, timestamp
        test_dataloader = DataLoader(test_data,batch_size=config.validate_batchsize, shuffle=False) #batch, batchsize, timestamp
        
        model, training_loss, validation_loss = train(args,train_dataloader=train_dataloader,
                                            test_dataloader=test_dataloader, 
                                            model=model, 
                                            train_epochs=train_epochs,
                                            early_stopping=early_stopping,
                                            path=path,
                                            writer= writer,
                                            )
        # writer.add_figure("Training Loss", visualize_data(np.asarray(training_loss), "Epochs", "Training Loss"))
        # writer.add_figure("Validation Loss", visualize_data(np.asarray(validation_loss), "Epochs", "Validation Loss"))
        
        writer.close()

        
    elif args.test:
        _, _, ctest_x, ctest_y  = train_target_split(c,days=config.seq_len, ratio=config.ratio, pred_len=config.pred_len) 
        test_data = TensorDataset(ctest_x,ctest_y)
        test_dataloader = DataLoader(test_data,batch_size=config.validate_batchsize, shuffle=False) #batch, batchsize, timestamp
        criterion = nn.MSELoss()
        best_model_path = path + '/' + 'checkpoint.pth'
        model.load_state_dict(torch.load(best_model_path))
        valid_loss = validate(args, test_dataloader=test_dataloader, 
                              model=model, 
                              criterion=criterion
                              )
        logging.warning("Validation Loss: {0:.7f}".format(
                valid_loss))
'''    
    elif args.visualize:
        writer.add_figure("Original Data",visualize_data(c.numpy(), "Timestamp", "Close Price"))
        writer.close()
        

            

        

        # writer.close()'''
        
    


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="description")
    parser.add_argument('--train', action='store_true')
    parser.set_defaults(train=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--m3', action='store_true')
    parser.set_defaults(m3 = False)
    parser.add_argument('--timemixer', action='store_true')
    parser.set_defaults(timemixer=False)
    parser.add_argument('--save_path', type=str, default='./', help='Directory to save checkpoint files')

    
    args = parser.parse_args()
    print(args)
    main(args=args)
    


    


