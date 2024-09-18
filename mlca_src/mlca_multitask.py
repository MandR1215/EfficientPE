from functools import partial
from collections import defaultdict
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from lightning.pytorch.utilities.combined_loader import CombinedLoader
import numpy as np
from typing import List, Dict, Any, Tuple
from ca_networks.ca_layers import ca_activation_func, CALayerReLUProjected
from mlca_src.mlca_nn_pt import MLCA_NN
import copy
from ca_networks.main import Net, weights_init, compute_metrics
import logging

# multitask_soft_nn_parameters = {
#     'batch_size'       : 4,
#     'epochs'           : 271,
#     'num_hidden_layers': 2,
#     'num_hidden_units' : 21,
#     'layer_type'       : 'CALayerReLUProjected',
#     'lr'               : 2e-3,
#     'loss_func'        : 'F.mse_loss',
#     'optimizer'        : 'Adam',
#     'l2'               : 1e-08
# }

multitask_soft_nn_parameters = {
    'batch_size'        : 4, 
    'epochs'            : 271, 
    'l2'                : 4.2059600269240764e-08, 
    'loss_func'         : 'F.mse_loss', 
    'lr'                : 0.0020276544345434492, 
    'num_hidden_layers' : 1, 
    'optimizer'         : 'Adam', 
    'layer_type'        : 'CALayerReLUProjected', 
    'num_hidden_units'  : 21,
    'include_id'        : True,
    'share_first'       : False,
    'id_emb_dim'        : 4,
}

multitask_hard_nn_parameters = {
    'batch_size'       : 4,
    'epochs'           : 5,
    'num_hidden_layers': 2,
    'num_hidden_units' : 21,
    'layer_type'       : 'CALayerReLUProjected',
    'lr'               : 2e-3,
    'loss_func'        : 'F.mse_loss',
    'optimizer'        : 'Adam',
    'l2'               : 1e-08
}

def eval_multitask_nn(bidders:List[str], 
                      fitted_scaler:Any, 
                      local_scaling_factor:float, 
                      NN_parameters:Dict[str, Any], 
                      elicited_bids:Dict[str, Any]):
    
    nn_model = MultiTask_MLCA_NN(bidders=bidders, 
                                 bids=elicited_bids, 
                                 scaler=fitted_scaler, 
                                 local_scaling_factor=local_scaling_factor)
    
    nn_model.initialize_model(NN_parameters)

    train_logs = nn_model.fit(X_valids=None, Y_valids=None)
    models = nn_model.mlca_nns

    result = {}
    for bidder in bidders:
        result[bidder] = [models[bidder], train_logs[bidder]]

    return result

class NegativeReLU(nn.Module):
    def __init__(self):
        super(NegativeReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        return -self.relu(-x)


class MultiTask_MLCA_NN:
    def __init__(self, 
                 bidders:List[str],
                 bids:Dict[str, np.array],
                 scaler:Any,
                 local_scaling_factor:float):
        """
        bidders              : ['Bidder_0', ...]
        bids                 : {'Bidder_i': np.array}
        scalers              : MinMaxScaler
        local_scaling_factors: float
        """

        self.bidders = bidders
        
        self.M = bids[bidders[0]][0].shape[1]
        assert all(bids[bidder][0].shape[1] == self.M for bidder in bidders)  

        self.X_trains = {bidder: bids[bidder][0] for bidder in bidders}
        self.Y_trains = {bidder: bids[bidder][1] for bidder in bidders}

        self.X_valids = None
        self.Y_valids = None
        
        self.model_parameters = None

        self.scaler = scaler
        self.local_scaling_factor = local_scaling_factor

        self.model_parameters = None

    def initialize_model(self, model_parameters:Dict[str, Any]):
        """
        model_parameters: {'num_hidden_layers': .., 'num_hidden_units': .., ...}
        """
        self.model_parameters = model_parameters

    def fit(self, X_valids:Dict[str, Any]=None, Y_valids:Dict[str, Any]=None):
        # validation data
        self.X_valids = X_valids
        self.Y_valids = Y_valids

        # # local scaling
        # target_max = {bidder: 1.0 for bidder in self.bidders}
        # mx_target = -float('inf')
        # for bidder in self.bidders:
        #     if self.local_scaling_factor is not None:
        #         mx_target = max(mx_target, self.Y_trains[bidder].max() / self.local_scaling_factor)
        # target_max = {bidder: mx_target for bidder in self.bidders}

        # local scaling
        target_max = {bidder: 1.0 for bidder in self.bidders}
        for bidder in self.bidders:
            if self.local_scaling_factor is not None:
                target_max[bidder] = max(target_max[bidder], self.Y_trains[bidder].max() / self.local_scaling_factor)
        self.Y_trains = {bidder: Y_train / target_max[bidder] for bidder, Y_train in self.Y_trains.items()}

        # train dataset
        train_dataset = dict()
        for i, (X_train, Y_train) in enumerate(zip(self.X_trains.values(), self.Y_trains.values())):
            dataset = TensorDataset(torch.from_numpy(X_train.astype(np.float32)),
                                    torch.from_numpy(Y_train.reshape(-1).astype(np.float32)))
            
            train_dataset[self.bidders[i]] = dataset

        # train config
        config = {
            'batch_size'        : self.model_parameters['batch_size'],
            'epochs'            : self.model_parameters['epochs'],
            'num_hidden_layers' : self.model_parameters['num_hidden_layers'],
            'num_units'         : self.model_parameters['num_hidden_units'],
            'layer_type'        : self.model_parameters['layer_type'],
            'input_dim'         : self.M,
            'lr'                : self.model_parameters['lr'],
            'loss_func'         : self.model_parameters['loss_func'],
            'target_max'        : target_max,
            'optimizer'         : self.model_parameters['optimizer'],
            'l2'                : self.model_parameters['l2'],
            'ts'                : float(1),
            'include_id'        : self.model_parameters['include_id'],
            'id_emb_dim'        : self.model_parameters['id_emb_dim'],
            'share_first'       : self.model_parameters['share_first']
        }
        self.config = config
        epochs = config['epochs']

        model, logs = train_model(train_dataset=train_dataset, config=config)

        self.history = logs
        self.trained_model = model
        self.logs_per_bidder = {bidder: logs['metrics']['train'][epochs][bidder] for bidder in self.bidders}
        self.__create_mlca_nns()        

        for bidder in self.bidders:
            logging.debug('loss: {:.7f}, kt: {:.4f}, r: {:.4f}'.format(
                logs['metrics']['train'][epochs][bidder]['loss'],
                logs['metrics']['train'][epochs][bidder]['kendall_tau'],
                logs['metrics']['train'][epochs][bidder]['r']))
            
        # get loss infos
        return self.logs_per_bidder
    
    def __create_mlca_nns(self):
        self.mlca_nns = {}
        logging.info("------------------- MTL Params -------------------")
        for bidder in self.bidders:
            model = MLCA_NN(X_train=self.X_trains[bidder],
                            Y_train=self.Y_trains[bidder],
                            scaler=self.scaler,
                            local_scaling_factor=self.local_scaling_factor)
            model.initialize_model(model_parameters=self.model_parameters)
            model.model = self.trained_model.models[bidder]
            logging.info(f"{bidder=}")
            for (name, param) in model.model.named_parameters():
                with np.printoptions(threshold=np.inf):
                    logging.info(f"{name=}")
                    logging.info(f"{param=}")
            model.history = self.logs_per_bidder[bidder]
            dataset_info = copy.deepcopy(self.config)
            dataset_info['target_max'] = self.config['target_max'][bidder]
            model.dataset_info = dataset_info

            self.mlca_nns[bidder] = model

class IDNet(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 num_hidden_layers: int, 
                 num_units: int, 
                 layer_type: str, 
                 target_max: float,
                 id_emb_dim: int,
                 ts: float = 1.0):
        super(IDNet, self).__init__()

        if layer_type == 'PlainNN':
            fc_layer = torch.nn.Linear
            self.activation_funcs = [torch.relu for _ in range(num_hidden_layers)]
            self.output_activation_function = torch.relu
        else:
            fc_layer = eval(layer_type)
            if type(ts) == float or type(ts) == int:
                self.ts = [ts] * (num_hidden_layers)
            else:
                self.ts = ts
            self.activation_funcs = [partial(ca_activation_func, t=t) for t in self.ts]
            self.output_activation_function = torch.nn.Identity()

        self._fc_layer = fc_layer
        self._input_dim = input_dim
        self._layer_type = layer_type
        self._num_hidden_layers = num_hidden_layers
        self._layer_type = layer_type
        self._target_max = target_max
        self._num_units = num_units
        self._id_emb_dim = id_emb_dim

        self.layers = []
        fc1 = fc_layer(input_dim, num_units)
        self.layers.append(fc1)
        for i in range(num_hidden_layers - 1):
            if i == 0:
                self.layers.append(fc_layer(num_units+id_emb_dim, num_units))
            else:
                self.layers.append(fc_layer(num_units, num_units))
        self.layers = torch.nn.ModuleList(self.layers)
        if self._num_hidden_layers == 1:
            self.output_layer = fc_layer(num_units+id_emb_dim, 1) if layer_type == 'PlainNN' else fc_layer(num_units+id_emb_dim, 1, bias=False)
        else:
            self.output_layer = fc_layer(num_units, 1) if layer_type == 'PlainNN' else fc_layer(num_units, 1, bias=False)
        self.output_activation_function = F.relu if layer_type == 'PlainNN' else torch.nn.Identity()
        self.dataset_info = None
        assert len(self.layers) == len(
            self.activation_funcs), 'Incorrect number of layers and activation functions.'
        
        self.is_transformed = False
        
    def set_activation_functions(self, ts):
        assert len(self.layers) == len(ts), 'Incorrect number of layers and activation functions.'
        self.ts = ts
        self.activation_funcs = [partial(ca_activation_func, t=t) for t in ts]

    def forward(self, x, id_emb=None):
        assert id_emb is not None or self.is_transformed
        for i, (layer, activation_func) in enumerate(zip(self.layers, self.activation_funcs)):
            x = layer(x)
            x = activation_func(x)
            if not self.is_transformed and i == 0:
                x = torch.cat([x, id_emb], axis=1)

        # Output layer
        x = self.output_activation_function(self.output_layer(x))
        return x

    def transform_weights(self, trained_emb:torch.tensor):
        for layer in self.layers:
            if hasattr(layer, 'transform_weights'):
                layer.transform_weights()
        if hasattr(self.output_layer, 'transform_weights'):
            self.output_layer.transform_weights()

        renew_fc1 = self._fc_layer(self._input_dim, self._num_units+self._id_emb_dim)
        renew_fc1.weight.data[:self._num_units] = self.layers[0].weight.data
        renew_fc1.weight.data[self._num_units:] = 0.0
        renew_fc1.bias.data[:self._num_units] = self.layers[0].bias.data
        renew_fc1.bias.data[self._num_units:] = trained_emb.data
        self.layers[0] = renew_fc1

        self.is_transformed = True


class MultiTaskHardNet(nn.Module):
    def __init__(self, 
                 bidders: List[str],
                 config:Dict[str, Dict[str, Any]]) -> None:
        super(MultiTaskHardNet, self).__init__()
        
        self.bidders = bidders

        self.shared_layer = eval(config['layer_type'])(in_features=config['input_dim'], 
                                                       out_features=config['input_dim'])
        # self.shared_layer = eval(config['layer_type'])(in_features=config['input_dim'], 
        #                                                out_features=3)
        self.bidder_specific_models = nn.ModuleDict({
            bidder: Net(input_dim=config['input_dim'],
                        layer_type=config['layer_type'],
                        num_hidden_layers=config['num_hidden_layers'],
                        num_units=config['num_units'], 
                        target_max=config['target_max'],
                        ts=config['ts'])
            # bidder: Net(input_dim=3,
            #             layer_type=config['layer_type'],
            #             num_hidden_layers=config['num_hidden_layers'],
            #             num_units=config['num_units'], 
            #             target_max=config['target_max'][bidder],
            #             ts=config['ts'])
            for bidder in self.bidders
        })

        self.config = config


    def forward(self, bidder, x):
        x = self.shared_layer(x)
        output = self.bidder_specific_models[bidder].forward(x)

        return output

    def transform_weights(self):
        self.shared_layer.transform_weights()
        for model in self.bidder_specific_models.values():
            model.transform_weights()

    def create_bidder_models(self):
        models = {}
        for bidder in self.bidders:
            model = Net(input_dim=self.config['input_dim'],
                        layer_type=self.config['layer_type'],
                        num_hidden_layers=self.config['num_hidden_layers']+1,
                        num_units=self.config['num_units'], 
                        target_max=self.config['target_max'][bidder],
                        ts=self.config['ts'])

            model.layers[0] = copy.deepcopy(self.shared_layer)

            for h in range(self.config['num_hidden_layers']):
                model.layers[h+1] = self.bidder_specific_models[bidder].layers[h]
            model.output_layer = self.bidder_specific_models[bidder].output_layer

            models[bidder] = model

        self.models = models

    def soft_sharing_loss(self):
        return 0.0
    
class MultiTaskSoftNet(nn.Module):
    def __init__(self, 
                 bidders: List[str],
                 config:Dict[str, Dict[str, Any]]) -> None:
        super(MultiTaskSoftNet, self).__init__()
        
        self.bidders = bidders

        if config['include_id']:
            self.models = nn.ModuleDict({
                bidder: IDNet(input_dim=config['input_dim'],
                              layer_type=config['layer_type'],
                              num_hidden_layers=config['num_hidden_layers'],
                              num_units=config['num_units'], 
                              target_max=config['target_max'][bidder],
                              ts=config['ts'],
                              id_emb_dim=config['id_emb_dim'])
                for bidder in self.bidders
            })
            self.id_emb = nn.Sequential(
                nn.Embedding(num_embeddings=len(self.bidders), embedding_dim=config['id_emb_dim']),
                NegativeReLU()
            )
            self.bidder_id = {bidder: i for (i,bidder) in enumerate(self.bidders)}
        else:
            self.models = nn.ModuleDict({
                bidder: Net(input_dim=config['input_dim'],
                            layer_type=config['layer_type'],
                            num_hidden_layers=config['num_hidden_layers'],
                            num_units=config['num_units'], 
                            target_max=config['target_max'][bidder],
                            ts=config['ts'])
                for bidder in self.bidders
            })

        self.include_id = config['include_id']
        self.share_first = config['share_first']

    def soft_sharing_loss(self, lam:float=1.0):
        loss = 0.0

        for i in range(len(self.bidders)-1):
            model_i = self.models[self.bidders[i]]

            for j in range(i+1, len(self.bidders)):
                model_j = self.models[self.bidders[j]]

                if self.share_first:
                    layers_i, layers_j = model_i.layers[0], model_j.layers[0]
                else:
                    layers_i, layers_j = model_i.output_layer, model_j.output_layer

                for (name_i, parameter_i), (name_j, parameter_j) in zip(layers_i.named_parameters(), layers_j.named_parameters()):
                    if 'weight' in name_i and 'weight' in name_j:
                        loss += torch.norm(parameter_i - parameter_j, p=2)

        return lam * loss
    
    def set_activation_functions(self, bidder, ts):
        model = self.models[bidder]
        assert len(model.layers) == len(ts), 'Incorrect number of layers and activation functions.'
        model.ts = ts
        model.activation_funcs = [partial(ca_activation_func, t=t) for t in ts]

    def forward(self, bidder, x):
        if self.include_id:
            idx = torch.tensor([self.bidder_id[bidder]]*len(x))
            id_emb = self.id_emb(idx)
            output = self.models[bidder].forward(x, id_emb)
        else:
            output = self.models[bidder].forward(x)

        return output

    def transform_weights(self):
        for bidder, model in self.models.items():
            if self.include_id:
                idx = torch.tensor([self.bidder_id[bidder]])
                id_emb = self.id_emb(idx)
                model.transform_weights(trained_emb=id_emb)
            else:
                model.transform_weights()

    def create_bidder_models(self):
        pass

def train_model(train_dataset:Dict[str, TensorDataset], 
                config:Dict[str, Any], 
                val_dataset:Dict[str, TensorDataset]=None, 
                test_dataset:Dict[str, TensorDataset]=None, 
                log_path=None, 
                eval_test=False,
                save_datasets=False) -> Tuple[MultiTaskSoftNet, dict]:

    bidders = list(train_dataset.keys())

    model = MultiTaskSoftNet(bidders=bidders, config=config)
    # model = MultiTaskHardNet(bidders=bidders, config=config)
        
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['l2'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'], momentum=.9, weight_decay=config['l2'])
    else:
        raise ValueError

    loss_func = eval(config['loss_func'])
    iterables = {}
    for bidder in bidders:
        batch_size = config['batch_size']
        iterables[bidder] = DataLoader(train_dataset[bidder], 
                                       batch_size=batch_size, 
                                       shuffle=True)
    # iterables = {bidder: DataLoader(train_dataset[bidder], 
    #                                 batch_size=config['batch_size'], 
    #                                 shuffle=True)
    #             for bidder in bidders}
    train_loader = CombinedLoader(iterables, 'max_size')
    _ = iter(train_loader)

    # if val_dataset:
    #     val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=4096, num_workers=2)
    # if test_dataset:
    #     test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4096, num_workers=2)

    logs = defaultdict()
    metrics = defaultdict(dict)
    last_train_loss = np.inf
    best_model = None

    reattempt = True
    attempts = 0
    MAX_ATTEMPTS = 1
    while reattempt and attempts < MAX_ATTEMPTS:
        logging.info(attempts)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(config['epochs']))
        attempts += 1
        model.apply(weights_init)
        for epoch in range(1, config['epochs'] + 1):
            metrics['train'][epoch] = multitask_train(model, 
                                                      train_loader, 
                                                      optimizer, 
                                                      config,
                                                      loss_func=loss_func)
            scheduler.step()

        if last_train_loss > metrics['train'][epoch]['loss']:
            best_model = pickle.loads(pickle.dumps(model))
            last_train_loss = metrics['train'][epoch]['loss']

        if np.isnan(metrics['train'][epoch]['kendall_tau']) or metrics['train'][epoch]['kendall_tau'] < 0 or \
                metrics['train'][epoch]['r'] < 0.9:
            reattempt = True
        else:
            reattempt = False

    model = best_model
    # Transform the weights
    model.transform_weights()
    model.create_bidder_models()
    # if val_dataset is not None:
    #     metrics['val'][epoch] = test(model, device, val_loader, valid_true=True, plot=False, epoch=epoch,
    #                                  log_path=None, dataset_info=config, loss_func=loss_func)

    # if eval_test:
    #     if test_dataset is not None:
    #         metrics['test'][epoch] = test(model, device, test_loader, valid_true=False, epoch=epoch, plot=True,
    #                                       log_path=None, dataset_info=config, loss_func=loss_func)
    logs['metrics'] = metrics
    # if save_datasets:
    #     metrics['datasets'] = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset,
    #                            'target_max': config['target_max']}
    #     metrics['model'] = model
    # if log_path is not None and not save_datasets:
    #     json.dump(logs, open(os.path.join(log_path, 'results.json'), 'w'))
    return model, logs

def multitask_train(model:MultiTaskSoftNet, 
                    train_loader:CombinedLoader, 
                    optimizer, 
                    config:Dict,
                    loss_func):
    model.train()
    
    preds, targets = [], []
    preds_per_bidder, targets_per_bidder = defaultdict(lambda: list()), defaultdict(lambda: list())

    mertics = dict()

    loss_per_bidder = defaultdict(lambda: list())
    total_loss = 0
    total_loss_per_bidder = defaultdict(lambda: 0.0)

    total_len = 0
    for batch, _, _ in train_loader:
        optimizer.zero_grad()
        bidders = batch.keys()
        loss = 0
        for bidder in bidders:
            if batch[bidder] is not None:
                data, target = batch[bidder]
                output = model(bidder, data)

                l_output = output.detach().cpu().numpy().flatten().tolist()
                l_target = target.detach().cpu().numpy().flatten().tolist()

                preds.extend([o * config['target_max'][bidder] for o in l_output])
                targets.extend([t * config['target_max'][bidder] for t in l_target])

                preds_per_bidder[bidder].extend(l_output)
                targets_per_bidder[bidder].extend(l_target)

                bidder_loss = loss_func(output.flatten(), target.flatten())

                loss += bidder_loss

                loss_per_bidder[bidder].append(float(bidder_loss.detach().cpu()))
                total_loss_per_bidder[bidder] += float(bidder_loss.detach().cpu() * len(preds_per_bidder[bidder]))

                total_loss += float(bidder_loss.detach().cpu()) * len(preds_per_bidder[bidder])
                total_len += len(preds_per_bidder[bidder])

        print(f"{loss=}, {model.soft_sharing_loss()=}")
        loss += model.soft_sharing_loss(lam=1e-10)
        loss.backward()
        optimizer.step()

    metrics = {'loss': total_loss / total_len}
    # preds, targets = (np.array(preds) * config['target_max']).tolist(), \
    #                  (np.array(targets) * config['target_max']).tolist()
    metrics.update(compute_metrics(preds, targets))

    for bidder, output in preds_per_bidder.items():
        target = targets_per_bidder[bidder]

        output, target = (np.array(output) * config['target_max'][bidder]).tolist(), \
                         (np.array(target) * config['target_max'][bidder]).tolist()
        metrics_bidder = compute_metrics(output, target)
        metrics_bidder.update({'loss': total_loss_per_bidder[bidder] / len(train_loader.iterables[bidder].dataset)})

        metrics.update({bidder: metrics_bidder})

    return metrics
    

