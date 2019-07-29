import torch
import numpy as np
import os
import pickle
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm_
from model import SAHR
import pdb
import collections
import math
from torch import autograd
class Solver(object):

    def __init__(self, dataset, config):
        torch.manual_seed(2019)
        # Data loader
        self.config = config
        self.dataset = dataset
        self.embed_dim = config.embed_dim
        self.dataset_type = config.dataset_type
        # Training Configuration
        self.lr = config.lr
        self.batch_size = config.batch_size
        self.std = config.std
        self.l2_reg = config.l2_reg
        self.K=config.K
        self.num_epochs = config.num_epochs
        self.config = config
        self.model_save_dir = config.model_save_dir
        self.num_users = self.dataset.user_num
        self.num_items = self.dataset.item_num
        self.num_features = self.dataset.feature_dims
        self.num_user_features = len(self.dataset.user_features)
        self.num_item_features = len(self.dataset.item_features)
        self.epoch_step = 1
        self.evaluation_every_n_batchs = math.ceil(self.dataset.train_data['X'].shape[0] / self.batch_size)
        # Build model
        self.build_model()
        with open('actor_director_fm_emb.pkl', 'rb') as f:
            print('load feat_emb')
            [self.model.feat_embeddings,_,_,_,_,_] = pickle.load(f)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)
    def build_model(self):
        # Define SAHR
        self.model = SAHR(self.num_users,
                         self.num_items,
                         self.num_features,
                         self.num_user_features,
                         self.num_item_features,
                         std=self.std,
                         l2_reg=self.l2_reg,
                         embed_dim=self.embed_dim,
                         N=self.config.N,
                         d_model=self.config.d_model,
                         h=self.config.h,
                         d_ff=self.config.d_ff,
                         dropout=self.config.dropout,
                         d_ff2=[100, 64],
                         K=self.config.K)
        if torch.cuda.is_available():
            self.model.cuda()

        # Optimizers
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=0)

    def load_model(self):
        """Load the best SAHR model."""
        print('Loading the SAHR model')
        SAHR_path = os.path.join(self.model_save_dir, 'FM.ckpt')
        self.model.load_state_dict(torch.load(SAHR_path, map_location=lambda storage, loc: storage))

    def save_model(self):
        # Save model checkpoints
        torch.save(self.model.state_dict(), os.path.join(self.model_save_dir, 'best-SAHR_actor_director.ckpt'))

    def to_var(self, x, volatile=True):
        if torch.cuda.is_available():
            x = x.cuda()
        with torch.no_grad():
            x = Variable(x)
        return x
    def from_numpy(self, x):
        x = self.to_var(torch.from_numpy(x).type(torch.LongTensor))

        return x

    def train(self):
        print('train start')
        # sample some users to calculate recall validation
        prev_rmse = 2
        rmse = 1.9
        epoch = 1
        while epoch <= self.num_epochs:
            self.model.train()
            if rmse < prev_rmse:
                prev_rmse = rmse
                #self.save_model()
                '''
                with open('actor_director_fm_emb.pkl', 'wb') as f:
                    pickle.dump([self.model.feat_embeddings,
                                 self.model.user_embeddings,
                                 self.model.item_embeddings,
                                 self.model.user_bias,
                                 self.model.item_bias,
                                 self.model.global_bias], f)
                '''
#                print('Model saved')
            print('Epoch: {}'.format(epoch))
            # train model
            losses = []
            train_data = self.dataset.train_data['X']
            targets = self.dataset.train_data['Y']
            users = train_data[:,0]
            items = train_data[:,1]
            user_feats = train_data[:,2:2+self.num_user_features]
            item_feats = train_data[:,2+self.num_user_features:]
            user_feats_mask = (user_feats != -1).astype(np.int32)
            item_feats_mask = (item_feats != -1).astype(np.int32)
            pdb.set_trace()
            for i in range(self.evaluation_every_n_batchs):
                users_batch = users[i*self.batch_size:(i+1)*self.batch_size]
                items_batch = items[i*self.batch_size:(i+1)*self.batch_size]
                user_feats_batch = user_feats[i*self.batch_size:(i+1)*self.batch_size]
                user_feats_mask_batch = user_feats_mask[i*self.batch_size:(i+1)*self.batch_size]
                item_feats_batch = item_feats[i*self.batch_size:(i+1)*self.batch_size]
                item_feats_mask_batch = item_feats_mask[i*self.batch_size:(i+1)*self.batch_size]
                targets_batch = targets[i*self.batch_size:(i+1)*self.batch_size]
                users_batch = self.from_numpy(users_batch)
                items_batch = self.from_numpy(items_batch)
                if self.num_user_features > 0:
                    user_feats_batch = self.from_numpy(user_feats_batch)
                    user_feats_mask_batch = self.from_numpy(user_feats_mask_batch).unsqueeze(-2).type(torch.cuda.FloatTensor)
                item_feats_batch = self.from_numpy(item_feats_batch)
                item_feats_mask_batch = self.from_numpy(item_feats_mask_batch).unsqueeze(-2).type(torch.cuda.FloatTensor)
                targets_batch = self.from_numpy(targets_batch)
                self.model.zero_grad()
                #with autograd.detect_anomaly():
                loss = self.model(users_batch, items_batch, user_feats_batch, user_feats_mask_batch, item_feats_batch, item_feats_mask_batch, targets_batch)
                loss.backward()
                #clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()
                losses.append(loss.detach().cpu().numpy())
            torch.cuda.empty_cache()
            print("Training loss {}\n".format(np.mean(losses)))

            self.model.eval()
            if epoch % self.epoch_step == 0:
                val_data = self.dataset.test_data['X']
                targets = self.dataset.test_data['Y']
                users = val_data[:,0]
                items = val_data[:,1]
                user_feats = val_data[:,2:2+self.num_user_features]
                item_feats = val_data[:,2+self.num_user_features:]
                user_feats_mask = (user_feats != 0).astype(np.int32)
                item_feats_mask = (item_feats != 0).astype(np.int32)

                users = self.from_numpy(users)
                items = self.from_numpy(items)
                if self.num_user_features > 0:
                    user_feats = self.from_numpy(user_feats)
                    user_feats_mask = self.from_numpy(user_feats_mask).unsqueeze(-2).type(torch.cuda.FloatTensor)
                item_feats = self.from_numpy(item_feats)
                item_feats_mask = self.from_numpy(item_feats_mask).unsqueeze(-2).type(torch.cuda.FloatTensor)
                val_k = 5000
                validation_every_n_batchs = math.ceil(users.shape[0] / val_k)
                y_scores = []
                for i in range(validation_every_n_batchs):
                    users_batch = users[i*val_k:(i+1)*val_k]
                    items_batch = items[i*val_k:(i+1)*val_k]
                    user_feats_batch = user_feats[i*val_k:(i+1)*val_k]
                    user_feats_mask_batch = user_feats_mask[i*val_k:(i+1)*val_k]
                    item_feats_batch = item_feats[i*val_k:(i+1)*val_k]
                    item_feats_mask_batch = item_feats_mask[i*val_k:(i+1)*val_k]
                    scores = self.model.predict(users_batch, items_batch, user_feats_batch, user_feats_mask_batch, item_feats_batch, item_feats_mask_batch).squeeze()
                    y_scores += list(scores.cpu().data.numpy())
                y_scores = np.array(y_scores)
                rmse = np.mean(((y_scores - targets) ** 2))
                rmse = rmse ** 0.5
                mae = np.mean(np.abs(y_scores - targets))
                print("Rmse on (sampled) validation set: {}".format(rmse))
                print("mae on (sampled) validation set: {}".format(mae))
            epoch += 1
        torch.cuda.empty_cache()

