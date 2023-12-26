import torch
import numpy as np

criterion = torch.nn.L1Loss()

class Trainer:
    def __init__(self, model, opt, scaler=None, noise_std=0):
        self.model  = model
        self.opt    = opt
        self.scaler = scaler
        if noise_std > 0:
            self.noise = torch.distributions.Normal(0, noise_std)
        else:
            self.noise = None

    def train(self, dataloader):
        self.model.train()
        n_batch    = len(dataloader)
        train_loss = 0

        for features, target, _ in dataloader:
            scaled_target = self.scaler.scale(target) 
            if self.noise is not None:
                scaled_target += self.noise(scaled_target.shape).to(scaled_target.device)
            pred = self.model(features)
            loss = criterion(scaled_target, pred)

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            train_loss += loss.detach().item()
        return train_loss/n_batch

    def test(self, dataloader):
        self.model.eval()
        n_batch      = len(dataloader)
        valid_loss   = 0
        list_ids     = []
        list_targets = []
        list_preds   = []

        with torch.no_grad():
            for features, target, ids in dataloader:
                scaled_target = self.scaler.scale(target)
                pred = self.model(features)
                loss = criterion(scaled_target, pred)

                valid_loss += loss.detach().item()

                list_ids.append(ids.reshape(-1))
                list_targets.append(target)
                list_preds.append(self.scaler.restore(pred))
                
        ids = np.hstack(list_ids)
        targets = torch.concat(list_targets, dim=0).cpu().numpy()
        preds = torch.concat(list_preds, dim=0).cpu().numpy()
        
        return valid_loss/n_batch, ids, targets, preds

    def predict(self, dataloader):
        self.model.eval()

        list_ids   = []
        list_preds = []

        with torch.no_grad():
            for features, _, ids in dataloader:
                pred = self.model(features)

                list_ids.append(ids.reshape(-1))
                list_preds.append(self.scaler.restore(pred))

        ids = np.hstack(list_ids)
        preds = torch.concat(list_preds, dim=0).cpu().numpy()
        
        return ids, preds

class SSIBTrainer:
    def __init__(self, model, opt):
        self.model = model
        self.opt   = opt

    def train(self, dataloader):
        self.model.train()
        train_mi   = 0
        train_loss = 0
        n_batch    = len(dataloader)
        for features, _, _ in dataloader:
            mi = self.model(features)
            loss = self.model.loss
        
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        
            train_mi   += mi.detach().item()
            train_loss += loss.detach().item()
        return train_mi/n_batch, train_loss/n_batch

    def test(self, dataloader):
        self.model.eval()
        
        valid_mi   = 0
        valid_loss = 0
        scalars    = None
        
        n_batch = len(dataloader)
        with torch.no_grad():
            for features, _, _ in dataloader:
                mi   = self.model(features)
                loss = self.model.loss

                valid_mi   += mi.detach().item()
                valid_loss += loss.detach().item()

                if scalars is None:
                    scalars = {k:v.detach().item() / n_batch for k,v in self.model.scalars.items()}
                else:
                    for k,v in self.model.scalars.items():
                        scalars[k] += v.detach().item() / n_batch

        return valid_mi/n_batch, valid_loss/n_batch, scalars