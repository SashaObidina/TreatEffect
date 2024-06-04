import random
import torch
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import scipy.stats as stats
import math

from torch import nn
from ConfigSpace import ConfigurationSpace
from copy import deepcopy
from hyperparameter_classes import *

default_device = 'cuda:3' if torch.cuda.is_available() else 'cpu'


trunc_norm_sampler_f = lambda mu, sigma : lambda: stats.truncnorm((0 - mu) / sigma, (1000000 - mu) / sigma, loc=mu, scale=sigma).rvs(1)[0]
beta_sampler_f = lambda a, b : lambda : np.random.beta(a, b)
gamma_sampler_f = lambda a, b : lambda : np.random.gamma(a, b)
uniform_sampler_f = lambda a, b : lambda : np.random.uniform(a, b)
uniform_int_sampler_f = lambda a, b : lambda : round(np.random.uniform(a, b))


def torch_masked_mean(x, mask, dim=0, return_share_of_ignored_values=False):
    """
    Returns the mean of a torch tensor and only considers the elements, where the mask is true.
    If return_share_of_ignored_values is true it returns a second tensor with the percentage of ignored values
    because of the mask.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    if return_share_of_ignored_values:
        return value / num, 1.-num/x.shape[dim]
    return value / num

def torch_masked_std(x, mask, dim=0):
    """
    Returns the std of a torch tensor and only considers the elements, where the mask is true.
    If get_mean is true it returns as a first Tensor the mean and as a second tensor the std.
    """
    num = torch.where(mask, torch.full_like(x, 1), torch.full_like(x, 0)).sum(dim=dim)
    value = torch.where(mask, x, torch.full_like(x, 0)).sum(dim=dim)
    mean = value / num
    mean_broadcast = torch.repeat_interleave(mean.unsqueeze(dim), x.shape[dim], dim=dim)
    quadratic_difference_from_mean = torch.square(torch.where(mask, mean_broadcast - x, torch.full_like(x, 0)))
    return torch.sqrt(torch.sum(quadratic_difference_from_mean, dim=dim) / (num - 1))

def torch_nanmean(x, dim=0, return_nanshare=False):
    return torch_masked_mean(x, ~torch.isnan(x), dim=dim, return_share_of_ignored_values=return_nanshare)

def torch_nanstd(x, dim=0):
    return torch_masked_std(x, ~torch.isnan(x), dim=dim)

def normalize_data(data, normalize_positions=-1):
    if normalize_positions > 0:
        mean = torch_nanmean(data[:normalize_positions], dim=0)
        std = torch_nanstd(data[:normalize_positions], dim=0) + .000001
    else:
        mean = torch_nanmean(data, dim=0)
        std = torch_nanstd(data, dim=0) + .000001
    data = (data - mean) / std
    data = torch.clip(data, min=-100, max=100)

    return data


def get_general_config(max_features, bptt, eval_positions=None):
    """"
    Returns the general PFN training hyperparameters.
    """
    config_general = {
        "lr": CSH.UniformFloatHyperparameter('lr', lower=0.0001, upper=0.00015, log=True), # learning rate
        "dropout": CSH.CategoricalHyperparameter('dropout', [0.0]),
        "emsize": CSH.CategoricalHyperparameter('emsize', [2 ** i for i in range(8, 9)]), ## upper bound is -1
        "batch_size": CSH.CategoricalHyperparameter('batch_size', [2 ** i for i in range(6, 8)]),
        "nlayers": CSH.CategoricalHyperparameter('nlayers', [12]),
        "num_features": max_features,
        "nhead": CSH.CategoricalHyperparameter('nhead', [4]),
        "nhid_factor": 2,
        "bptt": bptt,
        "eval_positions": eval_positions,
        "seq_len_used": bptt,
        "sampling": 'normal',#hp.choice('sampling', ['mixed', 'normal']), # uniform
        "epochs": 80,
        "num_steps": 100,
        "verbose": True,
        "mix_activations": True,
        "pre_sample_causes": True,
        "multiclass_type": 'rank'
    }

    return config_general


def get_flexible_categorical_config(max_features):
    """"
    Returns the configuration parameters for the tabular multiclass wrapper.
    """
    config_flexible_categorical = {
        "nan_prob_unknown_reason_reason_prior": CSH.CategoricalHyperparameter('nan_prob_unknown_reason_reason_prior', [0.5]),
        "categorical_feature_p": CSH.CategoricalHyperparameter('categorical_feature_p', [0.0, 0.1, 0.2]),
        "nan_prob_no_reason": CSH.CategoricalHyperparameter('nan_prob_no_reason', [0.0, 0.1]),
        "nan_prob_unknown_reason": CSH.CategoricalHyperparameter('nan_prob_unknown_reason', [0.0]),
        "nan_prob_a_reason": CSH.CategoricalHyperparameter('nan_prob_a_reason', [0.0]),
        # "num_classes": lambda : random.randint(2, 10), "balanced": False,
        "max_num_classes": 2,
        "num_classes": 2,
        "noise_type": CSH.CategoricalHyperparameter('noise_type', ["Gaussian"]), # NN
        "balanced": True,
        "normalize_to_ranking": CSH.CategoricalHyperparameter('normalize_to_ranking', [False]),
        "set_value_to_nan": CSH.CategoricalHyperparameter('set_value_to_nan', [0.5, 0.2, 0.0]),
        "normalize_by_used_features": True,
        "num_features_used":
            {'uniform_int_sampler_f(3,max_features)': uniform_int_sampler_f(1, max_features)}
        # hp.choice('conv_activation', [{'distribution': 'uniform', 'min': 2.0, 'max': 8.0}, None]),
    }
    return config_flexible_categorical


def get_diff_causal():
    """"
    Returns the configuration parameters for a differentiable wrapper around MLP / Causal mixture.
    """
    diff_causal = {
        "num_layers": {'distribution': 'meta_gamma', 'max_alpha': 2, 'max_scale': 3, 'round': True,
                       'lower_bound': 2},
        "prior_mlp_hidden_dim": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 100, 'round': True, 'lower_bound': 4},

        "prior_mlp_dropout_prob": {'distribution': 'meta_beta', 'scale': 0.6, 'min': 0.1, 'max': 5.0},
    # This mustn't be too high since activations get too large otherwise

        "noise_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': .3, 'min_mean': 0.0001, 'round': False,
                      'lower_bound': 0.0},
        "init_std": {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10.0, 'min_mean': 0.01, 'round': False,
                     'lower_bound': 0.0},
        "num_causes": {'distribution': 'meta_gamma', 'max_alpha': 3, 'max_scale': 7, 'round': True,
                                 'lower_bound': 2},

        "is_causal": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "pre_sample_weights": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "y_is_effect": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sampling": {'distribution': 'meta_choice', 'choice_values': ['normal', 'mixed']},
        "prior_mlp_activations": {'distribution': 'meta_choice_mixed', 'choice_values': [
            torch.nn.Tanh
            , torch.nn.Identity
            , torch.nn.ReLU
        ]},
        "block_wise_dropout": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "sort_features": {'distribution': 'meta_choice', 'choice_values': [True, False]},
        "in_clique": {'distribution': 'meta_choice', 'choice_values': [True, False]},
    }

    return diff_causal


def get_diff_flex():
    """"
    Returns the configuration parameters for a differentiable wrapper around the tabular multiclass wrapper.
    """
    diff_flex = {
        # "ordinal_pct": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5},
        # "num_categorical_features_sampler_a": hp.choice('num_categorical_features_sampler_a',
        #                                                 [{'distribution': 'uniform', 'min': 0.3, 'max': 0.9}, None]),
        # "num_categorical_features_sampler_b": {'distribution': 'uniform', 'min': 0.3, 'max': 0.9},

        "output_multiclass_ordered_p": {'distribution': 'uniform', 'min': 0.0, 'max': 0.5}, #CSH.CategoricalHyperparameter('output_multiclass_ordered_p', [0.0, 0.1, 0.2]),
        "multiclass_type": {'distribution': 'meta_choice', 'choice_values': ['value', 'rank']},
    }

    return diff_flex


def get_diff_gp():
    """"
    Returns the configuration parameters for a differentiable wrapper around GP.
    """
    diff_gp = {
        'outputscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
                        'lower_bound': 0},
        'lengthscale': {'distribution': 'meta_trunc_norm_log_scaled', 'max_mean': 10., 'min_mean': 0.00001, 'round': False,
                        'lower_bound': 0},
        'noise': {'distribution': 'meta_choice', 'choice_values': [0.00001, 0.0001, 0.01]}
    }

    return diff_gp

def get_diff_prior_bag():
    """"
    Returns the configuration parameters for a GP and MLP / Causal mixture.
    """
    diff_prior_bag = {
        'prior_bag_exp_weights_1': {'distribution': 'uniform', 'min': 2.0, 'max': 10.0},
        # MLP Weight (Biased, since MLP works better, 1.0 is weight for prior number 0)
    }

    return diff_prior_bag

def get_diff_config():
    """"
    Returns the configuration parameters for a differentiable wrapper around GP and MLP / Causal mixture priors.
    """
    diff_prior_bag = get_diff_prior_bag()
    diff_causal = get_diff_causal()
    diff_gp = get_diff_gp()
    diff_flex = get_diff_flex()

    config_diff = {'differentiable_hyperparameters': {**diff_prior_bag, **diff_causal, **diff_gp, **diff_flex}}

    return config_diff


def get_prior_config_causal(max_features=100):
    config_general = get_general_config(max_features, 50, eval_positions=[30])
    config_general_real_world = {**config_general}

    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_flexible_categorical_real_world = {**config_flexible_categorical}
    config_flexible_categorical_real_world[
        'num_categorical_features_sampler_a'] = -1.0  # Categorical features disabled by default

    config_gp = {}
    config_mlp = {}

    config_diff = get_diff_config()

    config_general_real_world['task_type'] = 'regression'
    if 'multiclass_type' in config_general_real_world:
        del config_general_real_world['multiclass_type']

    config = {**config_general_real_world, **config_flexible_categorical_real_world, **config_diff, **config_gp,
              **config_mlp}

    return config


def get_prior_config_bnn(max_features=100):
    config_general = get_general_config(max_features, 50, eval_positions=[30])
    config_general_real_world = {**config_general}

    config_flexible_categorical = get_flexible_categorical_config(max_features)
    config_flexible_categorical_real_world = {**config_flexible_categorical}

    config_gp = {}
    config_mlp = {}

    config_diff = get_diff_config()

    config = {**config_general_real_world, **config_flexible_categorical_real_world, **config_diff, **config_gp,
              **config_mlp}

    config['differentiable_hyperparameters']['prior_bag_exp_weights_1'] = {'distribution': 'uniform',
                                                                                  'min': 1000.0,
                                                                                  'max': 1001.0}  # Always select MLP
    return config


def get_default_spec(test_datasets, valid_datasets):
    bptt = 10000
    eval_positions = [1000, 2000, 3000, 4000, 5000] # list(2 ** np.array([4, 5, 6, 7, 8, 9, 10, 11, 12]))
    max_features = max([X.shape[1] for (_, X, _, _, _, _) in test_datasets] + [X.shape[1] for (_, X, _, _, _, _) in valid_datasets])
    max_splits = 5

    return bptt, eval_positions, max_features, max_splits


def get_prior_config(config_type):
    if config_type == 'causal':
        return get_prior_config_causal()
    elif config_type == 'bnn':
        return get_prior_config_bnn()


def replace_differentiable_distributions(config):
    import ConfigSpace.hyperparameters as CSH
    diff_config = config['differentiable_hyperparameters']
    for name, diff_hp_dict in diff_config.items():
        distribution = diff_hp_dict['distribution']
        if distribution == 'uniform':
            diff_hp_dict['sample'] = CSH.UniformFloatHyperparameter(name, diff_hp_dict['min'], diff_hp_dict['max'])
        elif distribution == 'meta_beta':
            diff_hp_dict['k'] = CSH.UniformFloatHyperparameter(name+'_k', diff_hp_dict['min'], diff_hp_dict['max'])
            diff_hp_dict['b'] = CSH.UniformFloatHyperparameter(name+'_b', diff_hp_dict['min'], diff_hp_dict['max'])
        elif distribution == 'meta_gamma':
            diff_hp_dict['alpha'] = CSH.UniformFloatHyperparameter(name+'_k', 0.0, math.log(diff_hp_dict['max_alpha']))
            diff_hp_dict['scale'] = CSH.UniformFloatHyperparameter(name+'_b', 0.0, diff_hp_dict['max_scale'])
        elif distribution == 'meta_choice':
            for i in range(1, len(diff_hp_dict['choice_values'])):
                diff_hp_dict[f'choice_{i}_weight'] = CSH.UniformFloatHyperparameter(name+f'choice_{i}_weight', -3.0, 5.0)
        elif distribution == 'meta_choice_mixed':
            for i in range(1, len(diff_hp_dict['choice_values'])):
                diff_hp_dict[f'choice_{i}_weight'] = CSH.UniformFloatHyperparameter(name+f'choice_{i}_weight', -3.0, 5.0)
        elif distribution == 'meta_trunc_norm_log_scaled':
            diff_hp_dict['log_mean'] = CSH.UniformFloatHyperparameter(name+'_log_mean', math.log(diff_hp_dict['min_mean']), math.log(diff_hp_dict['max_mean']))
            min_std = diff_hp_dict['min_std'] if 'min_std' in diff_hp_dict else 0.1
            max_std = diff_hp_dict['max_std'] if 'max_std' in diff_hp_dict else 1.0
            diff_hp_dict['log_std'] = CSH.UniformFloatHyperparameter(name+'_log_std', math.log(min_std), math.log(max_std))
        else:
            raise ValueError(f'Unknown distribution {distribution}')


def list_all_hps_in_nested(config):
    """"
    Returns a list of hyperparameters from a neszed dict of hyperparameters.
    """

    if isinstance(config, CSH.Hyperparameter):
        return [config]
    elif isinstance(config, dict):
        result = []
        for k, v in config.items():
            result += list_all_hps_in_nested(v)
        return result
    else:
        return []

def create_configspace_from_hierarchical(config):
    cs = CS.ConfigurationSpace()
    for hp in list_all_hps_in_nested(config):
        cs.add_hyperparameter(hp)
    return cs

def fill_in_configsample(config, configsample):
    # config is our dict that defines config distribution
    # configsample is a CS.Configuration
    hierarchical_configsample = deepcopy(config)
    for k, v in config.items():
        if isinstance(v, CSH.Hyperparameter):
            hierarchical_configsample[k] = configsample[v.name]
        elif isinstance(v, dict):
            hierarchical_configsample[k] = fill_in_configsample(v, configsample)
    return hierarchical_configsample


def evaluate_hypers(config, sample_diff_hps=False):
    """"
    Samples a hyperparameter configuration from a sampleable configuration (can be used in HP search).
    """
    if sample_diff_hps:
        # I do a deepcopy here, such that the config stays the same and can still be used with diff. hps
        config = deepcopy(config)
        replace_differentiable_distributions(config)
    cs = create_configspace_from_hierarchical(config)
    cs_sample = cs.sample_configuration()
    return fill_in_configsample(config, cs_sample)


def sample_differentiable(config):
    """"
    Returns sampled hyperparameters from a differentiable wrapper, that is it makes a non-differentiable out of
    differentiable.
    """
    # config is a dict of dicts, dicts that have a 'distribution' key are treated as distributions to be sampled
    result = deepcopy(config)
    del result['differentiable_hyperparameters']

    for k, v in config['differentiable_hyperparameters'].items():
        s_indicator, s_hp = DifferentiableHyperparameter(**v, embedding_dim=None,
                                                         device=None)()  # both of these are actually not used to the best of my knowledge
        result[k] = s_hp

    return result


def unpack_dict_of_tuples(d):
    # Returns list of dicts where each dict i contains values of tuple position i
    # {'a': (1,2), 'b': (3,4)} -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 4}]
    return [dict(zip(d.keys(), v)) for v in list(zip(*list(d.values())))]


def get_batch_mlp(batch_size, seq_len, num_features, hyperparameters, device=default_device, num_outputs=1, sampling='normal'
              , epoch=None, **kwargs):
    if 'multiclass_type' in hyperparameters and hyperparameters['multiclass_type'] == 'multi_node':
        num_outputs = num_outputs * hyperparameters['num_classes']

    if not (('mix_activations' in hyperparameters) and hyperparameters['mix_activations']):
        s = hyperparameters['prior_mlp_activations']()
        hyperparameters['prior_mlp_activations'] = lambda : s

    class MLP(torch.nn.Module):
        def __init__(self, hyperparameters):
            super(MLP, self).__init__()

            with torch.no_grad():

                for key in hyperparameters:
                    setattr(self, key, hyperparameters[key] if not callable(hyperparameters[key]) else hyperparameters[key]())

                assert (self.num_layers >= 2)
                #print("num_layers", self.num_layers)

                #if 'verbose' in hyperparameters and self.verbose:
                '''print({k : hyperparameters[k] for k in ['is_causal', 'num_causes', 'prior_mlp_hidden_dim'
                    , 'num_layers', 'noise_std', 'y_is_effect', 'pre_sample_weights', 'prior_mlp_dropout_prob'
                    , 'pre_sample_causes']})'''

                if self.is_causal:
                    self.prior_mlp_hidden_dim = max(self.prior_mlp_hidden_dim, num_outputs + 2 * num_features)
                else:
                    self.num_causes = num_features

                # This means that the mean and standard deviation of each cause is determined in advance
                if self.pre_sample_causes:
                    self.causes_mean, self.causes_std = causes_sampler_f(self.num_causes)
                    self.causes_mean = torch.tensor(self.causes_mean, device=device).unsqueeze(0).unsqueeze(0).tile(
                        (seq_len, 1, 1))
                    self.causes_std = torch.tensor(self.causes_std, device=device).unsqueeze(0).unsqueeze(0).tile(
                        (seq_len, 1, 1))

                def generate_module(layer_idx, out_dim):
                    # Determine std of each noise term in initialization, so that is shared in runs
                    # torch.abs(torch.normal(torch.zeros((out_dim)), self.noise_std)) - Change std for each dimension?
                    noise = (GaussianNoise(torch.abs(torch.normal(torch.zeros(size=(1, out_dim), device=device), float(self.noise_std))), device=device)
                         if self.pre_sample_weights else GaussianNoise(float(self.noise_std), device=device))
                    return [
                        nn.Sequential(*[self.prior_mlp_activations()
                            , nn.Linear(self.prior_mlp_hidden_dim, out_dim)
                            , noise])
                    ]

                self.layers = [nn.Linear(self.num_causes, self.prior_mlp_hidden_dim, device=device)]
                self.layers += [module for layer_idx in range(self.num_layers-1) for module in generate_module(layer_idx, self.prior_mlp_hidden_dim)]
                if not self.is_causal:
                    self.layers += generate_module(-1, num_outputs)
                self.layers = nn.Sequential(*self.layers)

                # Initialize Model parameters
                for i, (n, p) in enumerate(self.layers.named_parameters()):
                    if self.block_wise_dropout:
                        if len(p.shape) == 2: # Only apply to weight matrices and not bias
                            nn.init.zeros_(p)
                            # TODO: N blocks should be a setting
                            n_blocks = random.randint(1, math.ceil(math.sqrt(min(p.shape[0], p.shape[1]))))
                            w, h = p.shape[0] // n_blocks, p.shape[1] // n_blocks
                            keep_prob = (n_blocks*w*h) / p.numel()
                            for block in range(0, n_blocks):
                                nn.init.normal_(p[w * block: w * (block+1), h * block: h * (block+1)], std=self.init_std / keep_prob)#**(1/2 if self.prior_mlp_scale_weights_sqrt else 1))
                    else:
                        if len(p.shape) == 2: # Only apply to weight matrices and not bias
                            dropout_prob = self.prior_mlp_dropout_prob if i > 0 else 0.0  # Don't apply dropout in first layer
                            dropout_prob = min(dropout_prob, 0.99)
                            nn.init.normal_(p, std=self.init_std / (1. - dropout_prob))#**(1/2 if self.prior_mlp_scale_weights_sqrt else 1)))
                            p *= torch.bernoulli(torch.zeros_like(p) + 1. - dropout_prob)

        def forward(self):
            def sample_normal():
                if self.pre_sample_causes:
                    causes = torch.normal(self.causes_mean, self.causes_std.abs()).float()
                else:
                    causes = torch.normal(0., 1., (seq_len, 1, self.num_causes), device=device).float()
                return causes

            if self.sampling == 'normal':
                causes = sample_normal()
            elif self.sampling == 'mixed':
                zipf_p, multi_p, normal_p = random.random() * 0.66, random.random() * 0.66, random.random() * 0.66
                def sample_cause(n):
                    if random.random() > normal_p:
                        if self.pre_sample_causes:
                            return torch.normal(self.causes_mean[:, :, n], self.causes_std[:, :, n].abs()).float()
                        else:
                            return torch.normal(0., 1., (seq_len, 1), device=device).float()
                    elif random.random() > multi_p:
                        x = torch.multinomial(torch.rand((random.randint(2, 10))), seq_len, replacement=True).to(device).unsqueeze(-1).float()
                        x = (x - torch.mean(x)) / torch.std(x)
                        return x
                    else:
                        x = torch.minimum(torch.tensor(np.random.zipf(2.0 + random.random() * 2, size=(seq_len)),
                                            device=device).unsqueeze(-1).float(), torch.tensor(10.0, device=device))
                        return x - torch.mean(x)
                causes = torch.cat([sample_cause(n).unsqueeze(-1) for n in range(self.num_causes)], -1)
            elif self.sampling == 'uniform':
                causes = torch.rand((seq_len, 1, self.num_causes), device=device)
            else:
                raise ValueError(f'Sampling is set to invalid setting: {sampling}.')

            outputs = [causes]
            for layer in self.layers:
                outputs.append(layer(outputs[-1]))
            outputs = outputs[2:]

            if self.is_causal:
                ## Sample nodes from graph if model is causal
                outputs_flat = torch.cat(outputs, -1)

                if self.in_clique:
                    random_perm = random.randint(0, outputs_flat.shape[-1] - num_outputs - num_features) + torch.randperm(num_outputs + num_features, device=device)
                else:
                    random_perm = torch.randperm(outputs_flat.shape[-1]-1, device=device)

                random_idx_y = list(range(-num_outputs, -0)) if self.y_is_effect else random_perm[0:num_outputs]
                random_idx = random_perm[num_outputs:num_outputs + num_features]

                if self.sort_features:
                    random_idx, _ = torch.sort(random_idx)
                y = outputs_flat[:, :, random_idx_y]

                x = outputs_flat[:, :, random_idx]
            else:
                y = outputs[-1][:, :, :]
                x = causes

            probability = random.uniform(0.4, 0.6) # t = 1 с вероятностью [0,4; 0,6]
            t = torch.bernoulli(torch.full_like(y, probability))

            if bool(torch.any(torch.isnan(x)).detach().cpu().numpy()) or bool(torch.any(torch.isnan(y)).detach().cpu().numpy()):
                print('Nan caught in MLP model x:', torch.isnan(x).sum(), ' y:', torch.isnan(y).sum())
                print({k: hyperparameters[k] for k in ['is_causal', 'num_causes', 'prior_mlp_hidden_dim'
                    , 'num_layers', 'noise_std', 'y_is_effect', 'pre_sample_weights', 'prior_mlp_dropout_prob'
                    , 'pre_sample_causes']})

                x[:] = 0.0
                y[:] = -100 # default ignore index for CE

            # random feature rotation
            if self.random_feature_rotation:
                x = x[..., (torch.arange(x.shape[-1], device=device)+random.randrange(x.shape[-1])) % x.shape[-1]]

            return x, y, t

    if hyperparameters.get('new_mlp_per_example', False):
        get_model = lambda: MLP(hyperparameters).to(device)
    else:
        model = MLP(hyperparameters).to(device)
        get_model = lambda: model

    sample = [get_model()() for _ in range(0, batch_size)]

    x, y, t = zip(*sample)
    y = torch.cat(y, 1).detach().squeeze(2)
    t = torch.cat(t, 1).detach().squeeze(2)
    x = torch.cat(x, 1).detach()

    return x, y, y, t


def get_mlp_prior_hyperparameters(config):
    #from tabpfn.priors.utils import gamma_sampler_f
    config = {hp: (list(config[hp].values())[0]) if type(config[hp]) is dict else config[hp] for hp in config}

    if 'random_feature_rotation' not in config:
        config['random_feature_rotation'] = True

    if "prior_sigma_gamma_k" in config:
        sigma_sampler = gamma_sampler_f(config["prior_sigma_gamma_k"], config["prior_sigma_gamma_theta"])
        config['init_std'] = sigma_sampler
    if "prior_noise_std_gamma_k" in config:
        noise_std_sampler = gamma_sampler_f(config["prior_noise_std_gamma_k"], config["prior_noise_std_gamma_theta"])
        config['noise_std'] = noise_std_sampler

    return config


@torch.no_grad()
def get_batch(batch_size, seq_len, num_features, get_batch
              , device=default_device, differentiable_hyperparameters={}
              , hyperparameters=None, batch_size_per_gp_sample=None, **kwargs):
    batch_size_per_gp_sample = batch_size_per_gp_sample or (min(64, batch_size)) # 1
    num_models = batch_size // batch_size_per_gp_sample # 512
    assert num_models * batch_size_per_gp_sample == batch_size, f'Batch size ({batch_size}) not divisible by batch_size_per_gp_sample ({batch_size_per_gp_sample})'

    args = {'device': device, 'seq_len': seq_len, 'num_features': num_features, 'batch_size': batch_size_per_gp_sample}
    args = {**kwargs, **args}

    models = [DifferentiablePrior(get_batch, hyperparameters, differentiable_hyperparameters, args) for _ in range(num_models)]
    sample = sum([[model()] for model in models], [])

    x, y, y_, t, hyperparameter_dict = zip(*sample)

    #if 'verbose' in hyperparameters and hyperparameters['verbose']:
        #print('Hparams', hyperparameter_dict[0].keys())

    hyperparameter_matrix = []
    for batch in hyperparameter_dict:
        hyperparameter_matrix.append([batch[hp] for hp in batch])

    transposed_hyperparameter_matrix = list(zip(*hyperparameter_matrix))
    assert all([all([hp is None for hp in hp_]) or all([hp is not None for hp in hp_]) for hp_ in transposed_hyperparameter_matrix]), 'it should always be the case that when a hyper-parameter is None, once it is always None'
    # we remove columns that are only None (i.e. not sampled)
    hyperparameter_matrix = [[hp for hp in hp_ if hp is not None] for hp_ in hyperparameter_matrix]
    if len(hyperparameter_matrix[0]) > 0:
        packed_hyperparameters = torch.tensor(hyperparameter_matrix)
        packed_hyperparameters = torch.repeat_interleave(packed_hyperparameters, repeats=batch_size_per_gp_sample, dim=0).detach()
    else:
        packed_hyperparameters = None

    x, y, y_, t, packed_hyperparameters = (torch.cat(x, 1).detach()
                                        , torch.cat(y, 1).detach()
                                        , torch.cat(y_, 1).detach()
                                        , torch.cat(t, 1).detach()
                                        , packed_hyperparameters) #list(itertools.chain.from_iterable(itertools.repeat(x, batch_size_per_gp_sample) for x in packed_hyperparameters)))#torch.repeat_interleave(torch.stack(packed_hyperparameters, 0).detach(), repeats=batch_size_per_gp_sample, dim=0))
    return x, y, y_, t, (packed_hyperparameters if hyperparameters.get('differentiable_hps_as_style', True) else None)


