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
from tqdm import tqdm

from data_provider import Data

import time
import os
import numpy as np

import shutil
import logging 
import pickle
import random


class EarlyStopping:
    def __init__(self, args, prev_counter, prev_best_loss, patience =7):
        self.patience = patience
        self.best_loss = None
        self.counter = 0
        self.val_loss_min = 0
        self.early_stop = False
        self.best = False
        if args.resume:
            self.load_prev(prev_counter=prev_counter, prev_best_loss=prev_best_loss)
    def __call__(self, epoch, optimizer, scheduler, train_loss,val_loss, model, path):
        score = -val_loss
        self.best = False
        if self.best_loss is None:
            self.best = True
            self.val_loss_min = val_loss
            self.best_loss = score
            path_best = path + '/' + 'bestmodel_checkpoint.pth'
            self.save_checkpoint(best= self.best, epoch=epoch, model = model, optimizer=optimizer, scheduler=scheduler, path=path_best,train_loss=train_loss, val_loss=val_loss, counter=self.counter)
            path_check = path + '/' + 'checkpoint.pth'
            self.save_checkpoint(self.best, epoch=epoch, model = model, optimizer=optimizer, scheduler=scheduler, path=path_check,train_loss=train_loss, val_loss=val_loss, counter=self.counter)
            

        elif score < self.best_loss:
            
            self.counter +=1
            print(f'Early Stopping counter {self.counter} out of {self.patience} ')
            if self.counter >= self.patience:
                self.early_stop = True
            path = path + '/' + 'checkpoint.pth'

            self.save_checkpoint(self.best, epoch=epoch, model = model, optimizer=optimizer, scheduler=scheduler,path=path,train_loss=train_loss, val_loss=val_loss, counter=self.counter)
        else:
            self.best = True
            self.best_loss = score
            path = path + '/' + 'bestmodel_checkpoint.pth'
            self.counter = 0
            self.save_checkpoint(self.best, epoch=epoch, model = model, optimizer=optimizer, scheduler=scheduler, path=path,train_loss=train_loss, val_loss=val_loss, counter=self.counter)
            
    '''def save_checkpoint(self, val_loss, model, path):
        print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    '''
    def save_checkpoint(self, best,epoch, model, optimizer, scheduler, path, train_loss, val_loss, counter):
        if best:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'counter' : counter,
        }, path)
        self.val_loss_min = val_loss
    def load_prev(self, prev_counter, prev_best_loss ):
        self.counter = prev_counter
        self.best_loss = prev_best_loss
        self.val_loss_min = prev_best_loss


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
    
        
def visualize_data(data, x_label, y_label):
    fig = plt.figure(figsize=(5,5))
    
    # ax = fig.add_subplot(1,2,1)
    plt.plot(data)
    # plt.set_title("Original data")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    return fig 


def load_best_model(path, model):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])

def load_checkpoint(path, model, optimizer, scheduler):
    
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1  # Resume from the next epoch
    training_loss = checkpoint['train_loss']
    validation_loss = checkpoint['val_loss']
    counter = checkpoint['counter']
    
    return start_epoch, training_loss, validation_loss, counter



def adjust_learning_rate(optimizer, scheduler, epoch):
    lr_adjust = {epoch: scheduler.get_last_lr()[0]}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        logging.warning(f'New learning rate: {lr}')
    return optimizer
    

def train(args, train_dataloader, test_dataloader, model, train_epochs, early_stopping, path, writer, lr, max_lr, weight_decay):
    global optimizer, scheduler
    criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    train_steps = len(train_dataloader)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
    #                                                 max_lr=0.0001, 
    #                                                 epochs=train_epochs, 
    #                                                 steps_per_epoch=train_steps)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, 
                                                    max_lr=max_lr, 
                                                    epochs=train_epochs, 
                                                    steps_per_epoch=train_steps)
    
    
    


    training_loss = []
    validation_loss = []
    start_epoch = 0
    
    
    model.train()
    if args.resume:
        start_epoch, train_loss, valid_loss, counter = load_checkpoint(path= str(path+'/'+'checkpoint.pth'),model=model,optimizer=optimizer ,scheduler=scheduler)
        print('Previous learning rate: ', optimizer.param_groups[0]['lr'])
        print('Previous scheduler: ', scheduler)
        print('Previous epoch:', str(start_epoch+1))
        print('Previous Training Loss: ', train_loss)
        print('Previous Validation Loss: ', valid_loss)
        print('Previous counter: ', counter)
        earlystopping= EarlyStopping(args=args,prev_counter=counter, prev_best_loss=valid_loss,patience=7)
        optimizer = adjust_learning_rate(optimizer=optimizer, scheduler= scheduler, epoch = start_epoch)
        scheduler.step()
    else:
        earlystopping = EarlyStopping(args=args, prev_best_loss=None, prev_counter=0, patience=7)
    time_now = time.time()
    
    for epoch in range(start_epoch, train_epochs):
        iter_count = 0 
        train_loss = []
        rmse_loss = []
        epoch_time = time.time()
        for i ,(data) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{train_epochs}")):
            iter_count += 1    
            optimizer.zero_grad()
            
            in_x, ground = torch.FloatTensor(data[0]).cuda(), torch.FloatTensor(data[1]).cuda()
            if args.tmkan:
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
            print(loss.item())
            train_loss.append(loss.item())
            rmse = torch.sqrt(loss)
            rmse_loss.append(rmse.item())
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
        # MSE Loss
        train_loss = np.average(train_loss)
        writer.add_scalar('MSE Training Loss', train_loss.item(), epoch + 1)
        training_loss.append([train_loss])
        valid_loss, rmse_valid_loss = validate(args, test_dataloader=test_dataloader, model=model, criterion=criterion)
        writer.add_scalar('MSE Validation Loss', valid_loss.item(), epoch + 1)
        validation_loss.append([valid_loss])

        #RMSE
        rmse_loss = np.average(rmse_loss)
        writer.add_scalar('RMSE Training Loss', rmse_loss.item(), epoch + 1)
        writer.add_scalar('RMSE Validation Loss', rmse_valid_loss.item(), epoch + 1)
        
        current_lr = optimizer.param_groups[0]['lr']
        writer.add_scalar('Learning Rate', current_lr, epoch + 1)

        logging.warning("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, valid_loss.item()))
        earlystopping(epoch = epoch,optimizer=optimizer,train_loss=train_loss.item(), scheduler = scheduler,val_loss=valid_loss.item(), model=model, path= path)
        if earlystopping.early_stop:
            logging.warning("Early Stopping")
            break
        optimizer = adjust_learning_rate(optimizer=optimizer, scheduler= scheduler, epoch = epoch+1 )
        scheduler.step()
        
        
            

        
        

    best_model_path = path + '/' + 'bestmodel_checkpoint.pth'
    
    # model.load_state_dict(torch.load(best_model_path))
    load_best_model(best_model_path, model=model)

    return model, training_loss, validation_loss



def validate(args, test_dataloader, model, criterion):
    valid_loss = []
    rmse_valid = []
    if args.save_result:
        pred_list = []
    model.eval()
    with torch.no_grad():
        for i, (data) in enumerate(tqdm(test_dataloader)):
            in_x, ground = torch.FloatTensor(data[0]).cuda(), torch.FloatTensor(data[1]).cuda()
            if args.tmkan:

                pred = []
                # pred, total_entropy = model(in_x)
                
                for a in range(in_x.shape[0]):
                    pred.append(model(in_x[a].unsqueeze(0)))
                pred = torch.cat(pred, dim=0)
                
            else:
                pred = model(in_x)
            pred = pred.detach().cpu()
            if args.save_result:
                pred_list.append(pred)
            ground = ground.detach().cpu()
            loss = criterion(pred,ground)
            
            valid_loss.append(loss.item())
            rmse = torch.sqrt(loss)
            rmse_valid.append(rmse.item())
            

    valid_loss = np.average(valid_loss)
    rmse_loss = np.average(rmse_valid)
    model.train()
    if args.save_result:
        return valid_loss, rmse_loss, pred_list
    return valid_loss, rmse_loss

def setup(args):
    if args.timemixer:
        path = os.path.join(args.save_path, 'model2_checkpoint')
        # path = "D:/ntu/stocks/model2_checkpoint"
        config = Model2Config()
        model = Model(configs=config).cuda()
        log_dir = os.path.join(args.save_path, 'model2')
        modelname = 'TimeMixer'        
    
    elif args.tmkan:
        path = os.path.join(args.save_path, 'model3_embedding3_checkpoint')
        # path = "D:/ntu/stocks/model3_embedding2_checkpoint"
        config = Model3Config()
        model = HybridModel(configs=config).cuda()
        log_dir = os.path.join(args.save_path, 'model3_embedding3')
        modelname = 'TimeMixer-Kan'
        
        # model.apply(init_weights)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)    
    if not os.path.exists(path):
        os.makedirs(path)
    print('model name: ', modelname)
    print('parameters count =',sum(p.numel() for p in model.parameters() if p.requires_grad))  
    
    return path, config, model, log_dir

def select_data(dataset, config):
    dataset = dataset[-(config.seq_len + config.pred_len + config.step):]
    return dataset

def PrepareData(args, dataset, config):
    if args.demo:
        dataset = select_data(dataset)
    #timestamp, features
    c = dataset.to_numpy()
    c = get_loader(c, dataset).squeeze(0)
    ctrain_x, ctrain_y, ctest_x, ctest_y  = train_target_split(c,days=config.seq_len, ratio=config.ratio, pred_len=config.pred_len) 
    train_data = TensorDataset(ctrain_x, ctrain_y) 
    test_data = TensorDataset(ctest_x,ctest_y)
    train_dataloader = DataLoader(train_data,batch_size=config.training_batchsize, shuffle=False) #batch, batchsize, timestamp
    test_dataloader = DataLoader(test_data,batch_size=config.validate_batchsize, shuffle=False) #batch, batchsize, timestamp
    return train_dataloader, test_dataloader    

def main(args):
    global model
    tqdm.pandas() 
    data = Data("AU8U.SI")
    dataset = data.extract_data(['Close','Open','High','Low'], tech_indicator=False)
    path, config, model, log_dir = setup(args)    
    torch.cuda.empty_cache()
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    if args.train:
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        train_dataloader, test_dataloader = PrepareData(args, dataset, config)
        model, training_loss, validation_loss = train(args,train_dataloader=train_dataloader,
                                            test_dataloader=test_dataloader, 
                                            model=model, 
                                            train_epochs=config.train_epoch,
                                            early_stopping=config.early_stopping,
                                            path=path,
                                            writer= writer,
                                            lr=config.lr,
                                            max_lr=config.max_lr,
                                            weight_decay = config.weight_decay
                                            )
        # writer.add_figure("Training Loss", visualize_data(np.asarray(training_loss), "Epochs", "Training Loss"))
        # writer.add_figure("Validation Loss", visualize_data(np.asarray(validation_loss), "Epochs", "Validation Loss"))
        
        writer.close()

    
    elif args.test:
        _, test_dataloader = PrepareData(args, dataset, config)
        criterion = nn.MSELoss()
        best_model_path = path + '/' + 'bestmodel_checkpoint.pth'
        load_best_model(best_model_path, model=model)
        # model.load_state_dict(torch.load(best_model_path))
        valid_loss, rmse_loss = validate(args, test_dataloader=test_dataloader, 
                              model=model, 
                              criterion=criterion
                              )
        logging.warning("Validation Loss: {0:.7f}".format(
                valid_loss))
        
    elif args.save_result:
        print('saving result')
        config.ratio = 1
        _, test_dataloader = PrepareData(args, dataset, config)
        criterion = nn.MSELoss()
        best_model_path = path + '/' + 'bestmodel_checkpoint.pth'
        load_best_model(best_model_path, model=model)
        # model.load_state_dict(torch.load(best_model_path))
        print('Running validation')
        valid_loss, rmse_loss, pred_list = validate(args, test_dataloader=test_dataloader, 
                              model=model, 
                              criterion=criterion
                              )
        logging.warning("Validation Loss: {0:.7f}".format(
                valid_loss))
        print('Saving predictions')

        with open(args.save_path + 'prediction.pkl', 'wb') as f:
            pickle.dump(pred_list, f)
        
        
        
        
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
    parser.add_argument('--resume', action='store_true')
    parser.set_defaults(resume=False)
    parser.add_argument('--test', action='store_true')
    parser.set_defaults(test=False)
    parser.add_argument('--visualize', action='store_true')
    parser.set_defaults(visualize=False)
    parser.add_argument('--tmkan', action='store_true')
    parser.set_defaults(tmkan = False)
    parser.add_argument('--timemixer', action='store_true')
    parser.set_defaults(timemixer=False)
    parser.add_argument('--save_result', action='store_true')
    parser.set_defaults(save_result=False)
    parser.add_argument('--demo', action= 'store_true')
    parser.set_defaults(demo = False)
    parser.add_argument('--save_path', type=str, default='./', help='Directory to save checkpoint files')
    
    
    args = parser.parse_args()
    print(args)
    main(args=args)
    


    


