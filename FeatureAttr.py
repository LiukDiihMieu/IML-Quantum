#!python3.9
# coding: utf-8

import torch
from scipy.special import comb
from itertools import combinations
from matplotlib.colors import CenteredNorm
from scipy.special import factorial
from numpy import log2
from tqdm import tqdm


#######################################################################
# feature attribution methods
#######################################################################


def vanilla_grad(x_test, model, target=None):
    
    '''
    Compute "saliency" or "vanilla gradient" explanation.
    
    v1.1 support batch input and is thus faster.
    
    Args:
        x_test: A target to be explained, allows a batch dimension.
        model: A predictor.
        target: A target class, by default the one predicted by model.
        
    Returns:
        A tensor of the same shape of x_test.
    '''
    
    if x_test.dim() == 1:  # automatic support of batch dimension
        x_test = x_test.unsqueeze(0)
    assert x_test.dim() == 2
    
    if target is None:  # automatic target
        target = model.pred(x_test)
    assert len(x_test) == len(target)
    
    x_test.requires_grad_()
    fx = model(x_test)
    target = fx[torch.arange(len(target)), target]  # integer array indexing
    target.sum().backward()
    
    # squeezed if no batch dimension in x_test
    x_grad = x_test.grad.squeeze()
    x_test.requires_grad_(False)
    
    return x_grad


def grad_x_input(x_test, model, target=None, vanilla=None, x0=None):
    
    '''
    Compute Gradient x Input explanation.
    
    In v1.3.1, it becomes actually "gradient times delta input", an
    arbitrary x0 is allow as a reference. In v1.3, we call vanilla_grad
    directly. Users are allowed to pass an extra parameter "vanilla" to
    take advantage of any computed vanilla gradient results. Doing so
    this function merely perform a multiplication.
    
    Args:
        x_test: A target to be explained, allows a batch dimension.
        model: A predictor.
        target: A target class, by default the one predicted by model.
        vanilla: Previously computed vanilla gradient. If supported,
        this function simply multiples it with x_test element-wisely.
        
    Returns:
        A tensor of the same shape as x_test.
    '''
    
    if x0 is None:
        x0 = torch.zeros_like(x_test)
    # If vanilla is not supported, compute from scratch.
    if vanilla is None:
        vanilla = vanilla_grad(x_test, model, target)
    
    return (x_test - x0) * vanilla


def integrated_grad(x_test, model, target=None, x0=None, m=30):
    '''
    Compute integrated graident.
    
    v1.2: allow batch x_test
    
    Args:
        x_test: A target to be explained, allows a batch dimension.
        model: A predictor.
        target: A target class, by default the one predicted by model.
        x0: A baseline.
        m: The number of steps used in numerical integration.
        
    Returns:
        A tensor of the same shape as x_test. May contains a batch
        dimension.
    '''
    
    m = m // 3 * 3
    
    if x_test.dim() == 1:
        x_test = x_test.unsqueeze(0)  # support batch dimension
    assert x_test.dim() == 2
    
    if x0 is None:
        x0 = torch.zeros_like(x_test)  # all zero vector as default baseline
    
    if target is None:  # automatic target
        target = model.pred(x_test)
    assert len(x_test) == len(target)

    # compute df/dx for all samples and all steps on those paths
    x_path = x0 + torch.einsum(
        # i:path, j:sample, k:feature
        'i,jk->ijk', torch.arange(0, m + 1).to(x_test.device) / m, x_test - x0)
    shape = [x_path.shape[0], x_path.shape[1], -1]
    x_path = x_path.reshape(-1, x_path.shape[-1])
    x_path.requires_grad_()  # x_path: (n_steps+1 x n_samples x n_features)
    # squeeze batch input to make it compatible with the model
    fx = model(x_path.squeeze()).reshape(shape)
    sum_trick = fx[:, torch.arange(len(target)), target].sum()
    # each sample and each step on the path independently contributes to the sum
    # first sum them up and backprop the sum to get all gradients from x_path
    sum_trick.backward()
    grad = x_path.grad.reshape(shape) # bring back the n_steps dimension
    
    # free up some memory
    del x_path, fx, sum_trick  
    torch.cuda.empty_cache()
    
    # compute numerical integral using Simpson's-3/8 formula
    simpson_coef = torch.tensor([1] + [3, 3, 2] * (m // 3 - 1) + [3, 3, 1]).float().to(x_test.device)
    ig = torch.einsum('ijk, i->jk', grad, simpson_coef)
    ig *= (x_test - x0) / m * 3 / 8
    
    # free up some memory
    del grad, simpson_coef
    torch.cuda.empty_cache()
    
    return ig


def smooth_grad(x_test, model, method, target=None, noise=.15, n=50, **kwargs):
    '''
    Compute "smooth gradient" variant of salicency maps.
    
    Args:
        x_test: A target to be explained, allows a batch dimension.
        model: A predictor.
        method: A feature attribution funuction.
        target: A target class, by default the one predicted by model.
        noise: strength of noise, normal distribution sigma 
        = noise * (extreme difference of x_test)
        n: number of samples to estimate the average
        kwargs: other key word arguments needed by method.
    
    Returns:
        A tensor of the same shape as x_test.
    '''
    
    if x_test.dim() == 1:
        x_test = x_test.unsqueeze(0)
    
    if target is None:
        target = model.pred(x_test)
    
    with torch.no_grad():
        sigma = noise * (x_test.max(1).values - x_test.min(1).values)
        noise = torch.randn(n, *x_test.shape).transpose(1, 2).to(x_test.device) * sigma
        x_noised = x_test + noise.transpose(1, 2)
        x_noised = x_noised.reshape(-1, x_test.shape[-1])
        t_noised = target.repeat(n)
    
    result = method(x_noised, model, t_noised, **kwargs)
    result = result.reshape(n, -1, x_test.shape[-1]).mean(0)
    
    # free up some memory
    del x_noised, t_noised
    torch.cuda.empty_cache()
    
    return result

#######################################################################
# attribution map plot
#######################################################################

def centered_attribution_plot(ax, attr_map, size=None, norm=None):
    if size is None:
        length = int(torch.sqrt(torch.tensor(attr_map.flatten().shape)))
        size = (length, length)
    
    im = ax.imshow(attr_map.reshape(size), cmap='coolwarm', norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    return im
    
    
#######################################################################
# Sensitivity-n benchmark
#######################################################################

def combine_vector(feature_indices, x, x0=None):
    '''
    Replace part of x by baseline value x0 according to indices
    specified by feature_indices.
    
    Args:
        feature_indices: A tensor specifying components to be replaced.
        x: A target. 
        x0: A baseline, by default a zero vector.
        
    Returns:
        A vector of the same shape as x.
    '''
    if x0 is None:
        x0 = torch.zeros_like(x)
    
    xx = x.detach().clone()
    xx[feature_indices] = x0[feature_indices]
    return xx

def feature_sampler(n_features, n_select, max_size=100):
    '''
    Return indices specifing combinations of a certain size.
    
    When the number of possible combination go beyond max_size, use
    sampling results instead of exhaustion.
    
    Args:
        n_features: total number of features
        n_select: subset size
        max_size: max number of subsets of dim n_select
        
    Returns:
        A list of tensor specifing indices of combinations.
    '''
    # small set: exhaust all combinations
    if comb(n_features, n_select) <= 100:
        return [torch.tensor(i) for i in combinations(range(n_features), n_select)]
    
    # large set, do sampling instead
    else:
        return [torch.randperm(n_features)[:n_select] for i in range(100)]
    
def subset_model_difference(model, feature_ind_list, x, x0=None, target_class=None):
    '''
    Compute the variance of model output after replacing a subset of
    features by baseline value.
    
    Args:
        model: A predictor model.
        feature_ind_list: A list of tensors specifing indices.
        x: A target.
        x0: A baseline.
        target_class: Target class that variance is computed.
        
    Returns:
        A list of variances of the same length as feature_ind_list.
    '''
    if x0 is None:
        x0 = torch.zeros_like(x)
    if target_class is None:
        target_class = model.pred(x.unsqueeze(0))
        
    x_rec = [combine_vector(f, x, x0) for f in feature_ind_list]
    x_rec = torch.stack(x_rec, dim=0)
    original_value = model(x.unsqueeze(0))[0]
    s_values = ((original_value - model(x_rec)).T[target_class]).squeeze()
    return s_values

def subset_attribution_map_sum(feature_ind_list, attribution_map):
    '''
    Sum over A certain subset of elements from an attribution map.
    
    Args:
        feature_ind_list: A list of tensors specifing indices.
        attribution_map: A tensor of an attribution map.
        
    Returns:
        A list of summations according to index list feeded.
    '''
    sum_of_selected = [attribution_map.squeeze()[f].sum()
                       for f in feature_ind_list]
    sum_of_selected = torch.stack(sum_of_selected, dim=0)
    return sum_of_selected
    
def sensitivity_n(model, n_max, attribution_map_list, x, x0=None, target_class=None):
    '''
    Compute sensitivity-n value for feature subset up to size n_max.
    
    It is optimized so that the computation is executed in an efficient way.
    
    Args:
        model: A predictor model.
        n_max: Maximum size of feature subset.
        attribution_map_list: A list of attribution maps corresponding
        to x, generated using different attribution methods.
        x: A target.
        x0: A baseline, by default a zero vector.
        target_class: Target class.
        
    Returns:
        A list of correlation coefficient lists for each attribution method
        in attribution_map_list.
    '''
    n_features = len(x)
    # store correlation
    corr = [[] for _ in range(len(attribution_map_list))]
    # store sums of feature map
    r_values = [[] for _ in range(len(attribution_map_list))]
    
    feature_indices = []  # store concatenated index list
    sample_length = []  # store index list lengths before concatenate
    
    # We concatenate feature subsets of distinct size and pass them
    # through the model only once, so that the parallel nature of the
    # model (if exists) can be maximally exploited.
    for n_select in range(1, n_max + 1):
        
        feature_indices_n = feature_sampler(n_features, n_select)
        sample_length.append(len(feature_indices_n))
        feature_indices += feature_indices_n
        
        # compute r_scores (no need to concatenate)
        for r_value, attribution_map in zip(r_values, attribution_map_list):
            r_value.append(subset_attribution_map_sum(feature_indices_n,
                                                      attribution_map))

    # use the concatenated feature indices list, call the model only once
    s_values = subset_model_difference(model, feature_indices, x, x0, 
                                       target_class=target_class)
    
    # split s_values into correct chunks
    it = iter(s_values)
    s_values = [torch.stack([next(it) for _ in range(l)], dim=0)
                for l in sample_length]
    
    # compute correlation
    for c, r_value in zip(corr, r_values):
        for rr, ss in zip(r_value, s_values):
            rr = rr.to(ss.device)
            t = torch.stack([rr, ss], dim=0)
            c.append(torch.corrcoef(t)[0, 1].item())
    return corr

#######################################################################
# Maximum-sensitivity benchmark
#######################################################################

def monte_carlo_neighborhood(x, radius, n):
    '''
    Performs Uniformly sampling within an n-dimensional ball.
    
    Uniform sampling with Muller method.
    
    Args:
        x: A target.
        radius: Defines the volume of neighborhood.
        n: Number of samples.
    Returns: 
        A tensor of samples of length n.
    '''
    d = len(x)
    u = torch.randn(n, d)
    r = torch.rand(n).pow(1 / d).unsqueeze(1)
    r = r * u / u.norm(dim=1).unsqueeze(1)
    r = r.to(x.device)
    return x + r * radius

def sensmax_monte_carlo(model, method, x, radius, target_class=None, x0=None, n=50, **kwargs):
    '''
    Maximize variance of attribution map by Monte-Carlo sampling.
    
    Args:
        model: A predictor model.
        method: A feature attribution funuction.
        x: A target.
        radius: Defines the volume of neighborhood.
        target_class: Target class.
        x0: A baseline, by default a zero vector.
        n: Number of Monte-Carlo samples.
        kwargs: other key word arguments needed by method.
        
    Returns:
        Max ||Phi(x + delta) - Phi(x)|| within the neighborhood of x
        of radius r. Phi is the feature attribution map selected, 
        ||...|| is the l-infinity norm.
    '''
    samples = monte_carlo_neighborhood(x, radius, n)
    attr_map = method(samples, model, target_class, **kwargs)
    attr_map0 = method(x, model, target_class, **kwargs)
    attr_map /= attr_map0.norm()
    attr_map0 /= attr_map0.norm()
    result = (attr_map - attr_map0).norm(dim=1, p=float('inf')).max().item()
    
    del samples, attr_map, attr_map0  # free up some memory
    torch.cuda.empty_cache()
    return result


def sensmax_monte_carlo(model, method, x, radius, target_class=None, x0=None, n=50, batch_size=50, **kwargs):
    '''
    Maximize variance of attribution map by Monte-Carlo sampling.
    
    Args:
        model: A predictor model.
        method: A feature attribution funuction.
        x: A target.
        radius: Defines the volume of neighborhood.
        target_class: Target class.
        x0: A baseline, by default a zero vector.
        n: Number of Monte-Carlo samples.
        batch_size: Number of samples to process in each batch.
        kwargs: other key word arguments needed by method.
        
    Returns:
        Max ||Phi(x + delta) - Phi(x)|| within the neighborhood of x
        of radius r. Phi is the feature attribution map selected, 
        ||...|| is the l-infinity norm.
    '''
    max_values = []
    num_batches = (n + batch_size - 1) // batch_size  # calculate the total number of batches
    attr_map0 = method(x, model, target_class, **kwargs)

    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, n)
        samples = monte_carlo_neighborhood(x, radius, end - start)
        attr_map = method(samples, model, target_class, **kwargs)
        attr_map /= attr_map0.norm()
        attr_map0 /= attr_map0.norm()
        result = (attr_map - attr_map0).norm(dim=1, p=float('inf')).max().item()
        max_values.append(result)
        del samples, attr_map  # free up some memory
        torch.cuda.empty_cache()

    return max(max_values)

#######################################################################
# Many shapley values
#######################################################################

def powerset(s):
    '''
    Returns all subsets of a list s.
    
    Args:
        s: iterable, full set.
    
    Yield:
        An iterable of all subsets, each as a sublist.
    '''
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask]

def subset_index(subset):
    '''
    Index of a subset: (0,1,4) --> 10011 --> 19.
    
    Args:
        subset: tuple or list
        
    Returns:
        An integer index.
    '''
    return (2 ** torch.tensor(subset)).sum().long()

def shapley(i, value_tensor):
    '''
    Returns the shapley value for feature i given the value function.
    
    Args:
        i: index for the feature
        value_tensor: pre-calculated value function tensor for all
        combinations of features.
        
    Returns:
        A float, shapley value for feature i.
    '''
    
    N = log2(len(value_tensor)).astype('int')  # num of features
    assert i < N, f"value_tensor only contains feature from 0 to {N-1}"
    
    players_without_i = list(range(N))
    players_without_i.pop(i)  # drop i
    
    subset_without = list(powerset(players_without_i))
    subset_with = [list(s) + [i] for s in subset_without]  # bring i back
    
    shap = 0
    for si, s in zip(subset_with, subset_without):
        vsi = v[subset_index(si)]  # v(S U i)
        vs = v[subset_index(s)]  # v(S)
        ns = len(s)  # size of set S
        coef = factorial(ns) * factorial(N - ns - 1) / factorial(N)
        shap += coef * (vsi - vs)
    return shap.item()

def baseline_value_function(x_test, model, x0=None):
    '''
    Compute value function for all possible combinations of features.
    
    Args:
        x_test: A target to be explained, no batch dimension.
        model: corresponding to the value function
        x0: A baseline, by default a zero vector.
        
    Returns:
        value tensor indexing by subset indices consistant with those
        generated by subset_index.
    '''
    if x0 is None:
        x0 = torch.zeros_like(x_test)
    
    N = len(x0)
    
    all_subsets = list(powerset(list(range(N))))
    x_rec = [combine_vector(f, x, x0) for f in tqdm(all_subsets)]
    x_rec = torch.stack(x_rec, dim=0)  # batch tensor
    value_tensor = model(x_rec)
    
    return value_tensor

def baseline_shap(x_test, model, x0=None):
    '''
    Compute BaselineSHAP.
    
    Args:
        x_test: A target to be explained, no batch dimension.
        model: corresponding to the value function
        x0: A baseline, by default a zero vector.
        
    Returns:
        Tensor of the same size as x_test, BSHAP for each feature.
    '''
    value_tensor = baseline_value_function(x_test, model, x0)
    shap = [shapley(i, value_tensor) for i in len(x0)]
    return torch.tensor(shap)
