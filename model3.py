import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import DataEmbedding, series_decomp
from StandardNorm import Normalize
from config import Model3Config

# B Spline
def B_batch(x, grid, k=0, extend=True, device='cpu'):
    '''
    evaludate x on B-spline bases

    Args:
    -----
        x : 2D torch.tensor
            inputs, shape (number of splines, number of samples)
        grid : 2D torch.tensor
            grids, shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        extend : bool
            If True, k points are extended on both ends. If False, no extension (zero boundary condition). Default: True
        device : str
            devicde

    Returns:
    --------
        spline values : 3D torch.tensor
            shape (number of splines, number of B-spline bases (coeffcients), number of samples). The numbef of B-spline bases = number of grid points + k - 1.

    Example
    -------
    >>>num_spline = 5
    >>>num_sample = 100
    >>>num_grid_interval = 10
    >>>k = 3
    >>>x = torch.normal(0,1,size=(num_spline, num_sample))
    >>>grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>>B_batch(x, grids, k=k).shape
    torch.Size([5, 13, 100])
    '''

    # x shape: (size, x); grid shape: (size, grid)
    def extend_grid(grid, k_extend=0):
        # pad k to left and right
        # grid shape: (batch, grid)
        h = (grid[:,  [-1]] - grid[:,  [0]]) / (grid.shape[1] - 1)

        for i in range(k_extend):
            grid = torch.cat([grid[:,  [0]] - h, grid], dim=1)
            grid = torch.cat([grid, grid[:,  [-1]] + h], dim=1)
        grid = grid.to(device)
        return grid

    if extend == True:
        grid = extend_grid(grid, k_extend=k)

    grid = grid.unsqueeze(dim=2).to(device)
    x = x.unsqueeze(dim=1).to(device)
    if k == 0:
        value = (x >= grid[:,  :-1]) * (x < grid[:,  1:])
    else:
        B_km1 = B_batch(x[:,  0], grid=grid[:, :,  0], k=k - 1, extend=False, device=device)
        
        value = (x - grid[:,  :-(k + 1)]) / (grid[:,  k:-1] - grid[:,  :-(k + 1)]) * B_km1[:,  :-1] + (
                grid[:,  k + 1:] - x) / (grid[:,  k + 1:] - grid[:,  1:(-k)]) * B_km1[:,  1:]
    

    
    # def extend_grid(grid, k_extend=0):
    #     # pad k to left and right
    #     # grid shape: (batch, grid)
    #     h = (grid[:, :, [-1]] - grid[:, :, [0]]) / (grid.shape[2] - 1)

    #     for i in range(k_extend):
    #         grid = torch.cat([grid[:, :, [0]] - h, grid], dim=2)
    #         grid = torch.cat([grid, grid[:, :, [-1]] + h], dim=2)
    #     grid = grid.to(device)
    #     return grid

    # if extend == True:
    #     grid = extend_grid(grid, k_extend=k)

    # grid = grid.unsqueeze(dim=3).to(device)
    # x = x.unsqueeze(dim=2).to(device)
    
    # if k == 0:
    #     value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    # else:

    #     B_km1 = B_batch(x[:, :, 0], grid=grid[:, :, :, 0], k=k - 1, extend=False, device=device)
        
    #     value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + (
    #             grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
     
    return value


def coef2curve(x_eval, grid, coef, k, device="cpu"):
    
    '''
    converting B-spline coefficients to B-spline curves. Evaluate x on B-spline curves (summing up B_batch results over B-spline basis).

    Args:
    -----
        x_eval : 2D torch.tensor)
            shape (number of splines, number of samples)
        grid : 2D torch.tensor)
            shape (number of splines, number of grid points)
        coef : 2D torch.tensor)
            shape (number of splines, number of coef params). number of coef params = number of grid intervals + k
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Returns:
    --------
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)

    Example
    -------
    >>>num_spline = 5
    >>>num_sample = 100
    >>>num_grid_interval = 10
    >>>k = 3
    >>>x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>>grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    >>>coef = torch.normal(0,1,size=(num_spline, num_grid_interval+k))
    >>>coef2curve(x_eval, grids, coef, k=k).shape
    torch.Size([5, 100])
    '''
    
    if coef.dtype != x_eval.dtype:
        coef = coef.to(x_eval.dtype)
    y_eval = torch.einsum('ij,ijk->ik', coef, B_batch(x_eval, grid, k, device=device))
    
    # y_eval = torch.einsum('ijn,ijmk->inmk', coef, B_batch(x_eval, grid, k, device=device))
    # y_eval = y_eval.reshape(y_eval.shape[0], -1, y_eval.shape[-1])
    return y_eval


def curve2coef(x_eval, y_eval, grid, k, device="cpu"):
    
    '''
    converting B-spline curves to B-spline coefficients using least squares.

    Args:
    -----
        x_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        y_eval : 2D torch.tensor
            shape (number of splines, number of samples)
        grid : 2D torch.tensor
            shape (number of splines, number of grid points)
        k : int
            the piecewise polynomial order of splines.
        device : str
            devicde

    Example
    -------
    >>>num_spline = 5
    >>>num_sample = 100
    >>>num_grid_interval = 10
    >>>k = 3
    >>>x_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>>y_eval = torch.normal(0,1,size=(num_spline, num_sample))
    >>>grids = torch.einsum('i,j->ij', torch.ones(num_spline,), torch.linspace(-1,1,steps=num_grid_interval+1))
    torch.Size([5, 13])
    '''
    # x_eval: (size, batch); y_eval: (size, batch); grid: (size, grid); k: scalar

    mat = B_batch(x_eval, grid, k, device=device).permute(0,2,1)
    
    # # dimension,  number of splines, grid points // dimension, splines, grid points 
    # mat = B_batch(x_eval, grid, k, device=device).permute(0,1,3,2)
    
    if(len(y_eval.shape) != 3):
        y_eval = y_eval.unsqueeze(dim=2)
    coef = torch.linalg.lstsq(mat.to(device), y_eval.to(device),
                              driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,0]
    
    # # coef = torch.linalg.lstsq(mat, y_eval.unsqueeze(dim=2)).solution[:, :, 0]
    # if(len(y_eval.shape) != 4):
    #     y_eval = y_eval.unsqueeze(dim=3)
    # coef = torch.linalg.lstsq(mat.to(device), y_eval.to(device),
    #                           driver='gelsy' if device == 'cpu' else 'gels').solution[:,:,0]
    return coef.to(device)

# KAN Layer
class KANLayer(nn.Module):
    """
    KANLayer class


    Attributes:
    -----------
        in_dim: int
            input dimension
        out_dim: int
            output dimension
        size: int
            the number of splines = input dimension * output dimension
        k: int
            the piecewise polynomial order of splines
        grid: 2D torch.float
            grid points
        noises: 2D torch.float
            injected noises to splines at initialization (to break degeneracy)
        coef: 2D torch.tensor
            coefficients of B-spline bases
        scale_base: 1D torch.float
            magnitude of the residual function b(x)
        scale_sp: 1D torch.float
            mangitude of the spline function spline(x)
        base_fun: fun
            residual function b(x)
        mask: 1D torch.float
            mask of spline functions. setting some element of the mask to zero means setting the corresponding activation to zero function.
        grid_eps: float in [0,1]
            a hyperparameter used in update_grid_from_samples. When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes.
        weight_sharing: 1D tensor int
            allow spline activations to share parameters
        lock_counter: int
            counter how many activation functions are locked (weight sharing)
        lock_id: 1D torch.int
            the id of activation functions that are locked
        device: str
            device

    Methods:
    --------
        __init__():
            initialize a KANLayer
        forward():
            forward
        update_grid_from_samples():
            update grids based on samples' incoming activations
        initialize_grid_from_parent():
            initialize grids from another model
        get_subset():
            get subset of the KANLayer (used for pruning)
        lock():
            lock several activation functions to share parameters
        unlock():
            unlock already locked activation functions
    """

    def __init__(self, in_dim=3, out_dim=2, d_model = 10, num=5, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True,
                 device='cuda'):
        ''''
        initialize a KANLayer

        Args:
        -----
            in_dim : int
                input dimension. Default: 2.
            out_dim : int
                output dimension. Default: 3.
            num : int
                the number of grid intervals = G. Default: 5.
            k : int
                the order of piecewise polynomial. Default: 3.
            noise_scale : float
                the scale of noise injected at initialization. Default: 0.1.
            scale_base : float
                the scale of the residual function b(x). Default: 1.0.
            scale_sp : float
                the scale of the base function spline(x). Default: 1.0.
            base_fun : function
                residual function b(x). Default: torch.nn.SiLU()
            grid_eps : float
                When grid_eps = 0, the grid is uniform; when grid_eps = 1, the grid is partitioned using percentiles of samples. 0 < grid_eps < 1 interpolates between the two extremes. Default: 0.02.
            grid_range : list/np.array of shape (2,)
                setting the range of grids. Default: [-1,1].
            sp_trainable : bool
                If true, scale_sp is trainable. Default: True.
            sb_trainable : bool
                If true, scale_base is trainable. Default: True.
            device : str
                device

        Returns:
        --------
            self

        Example
        -------
        >>>model = KANLayer(in_dim=3, out_dim=5)
        >>>(model.in_dim, model.out_dim)
        (3, 5)
        '''
        super(KANLayer, self).__init__()
        # size
        self.size = size = out_dim * in_dim
        self.out_dim = out_dim
        self.in_dim = in_dim
        self.num = num
        self.k = k

        # shape: (size, num)
        self.grid = torch.einsum('i,k->ik', torch.ones(size, device=device), 
                                 torch.linspace(grid_range[0], grid_range[1], steps=num + 1, device=device))
        
        self.grid = self.grid.unsqueeze(0).repeat(d_model,1,1)
        
        self.grid = torch.nn.Parameter(self.grid).requires_grad_(False)
        
        # noises = (torch.rand(size, self.grid.shape[1]) - 1 / 2) * noise_scale / num
        
        noises = (torch.rand(self.grid.shape[0] , self.size, self.grid.shape[2]) - 1 / 2) * noise_scale / num
        noises = noises.to(device)
        
        # shape: (size, coef)
        
        self.coef = torch.nn.Parameter(torch.cat([curve2coef(self.grid[i], noises[i], self.grid[i], k, device).unsqueeze(0) for i in range(d_model)], dim = 0))
        
        if isinstance(scale_base, float):
            # self.scale_base = torch.nn.Parameter(torch.ones(size, device=device) * scale_base).requires_grad_(
            #     sb_trainable)  # make scale trainable
            
            self.scale_base = torch.nn.Parameter(torch.ones(size*d_model, device=device).reshape(d_model, size) * scale_base).requires_grad_(
                sb_trainable)  # make scale trainable
        else:
            self.scale_base = torch.nn.Parameter(torch.FloatTensor(scale_base).to(device)).requires_grad_(sb_trainable)
        # self.scale_sp = torch.nn.Parameter(torch.ones(size, device=device) * scale_sp).requires_grad_(
        #     sp_trainable)  # make scale trainable
        
        self.scale_sp = torch.nn.Parameter(torch.ones(size*d_model, device=device).reshape(d_model, size) * scale_sp).requires_grad_(
            sp_trainable)  # make scale trainable
        self.base_fun = base_fun

        # self.mask = torch.nn.Parameter(torch.ones(size, device=device)).requires_grad_(False)
        
        self.mask = torch.nn.Parameter(torch.ones(size*d_model, device=device).reshape(d_model, size)).requires_grad_(False)
        self.grid_eps = grid_eps
        # self.weight_sharing = torch.arange(size)
        
        self.weight_sharing = torch.arange(size)
        self.lock_counter = 0
        # self.lock_id = torch.zeros(size)
        
        self.lock_id = torch.zeros(size*d_model).reshape(d_model, size)
        self.device = device

    def forward(self, x):
        '''
        KANLayer forward given input x

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            y : 2D torch.float
                outputs, shape (number of samples, output dimension)
            preacts : 3D torch.float
                fan out x into activations, shape (number of sampels, output dimension, input dimension)
            postacts : 3D torch.float
                the outputs of activation functions with preacts as inputs
            postspline : 3D torch.float
                the outputs of spline functions with preacts as inputs

        Example
        -------
        >>>model = KANLayer(in_dim=3, out_dim=5)
        >>>x = torch.normal(0,1,size=(100,3))
        >>>y, preacts, postacts, postspline = model(x)
        >>>y.shape, preacts.shape, postacts.shape, postspline.shape
        (torch.Size([100, 5]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]),
         torch.Size([100, 5, 3]))
        '''
        
        x = x.unsqueeze(1)
        dimension, batch = x.shape[0], x.shape[1]
        # x: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        
        x = torch.einsum('dij,k->dikj', x, torch.ones(self.out_dim, device=self.device)).reshape(dimension, batch,
                                                                                               self.size).permute(0, 2, 1)
        
        preacts = x.permute(0 ,2, 1).clone().reshape(dimension, batch, self.out_dim, self.in_dim)
        base = self.base_fun(x).permute(0, 2, 1)  # shape (batch, size)
        y = torch.cat([coef2curve(x_eval=x[i], grid = self.grid[i,self.weight_sharing,:], coef = self.coef[i,self.weight_sharing,:], k = self.k, device=self.device).unsqueeze(0) for i in range(dimension)],dim=0)
        # y = coef2curve(x_eval=x, grid=self.grid[:,self.weight_sharing,:], coef=self.coef[:,self.weight_sharing,:], k=self.k,
        #                device=self.device)  # shape (size, batch)
        
        y = y.permute(0,2,1)  # shape (batch, size)
        postspline = y.clone().reshape(dimension, batch, self.out_dim, self.in_dim)
        
        y = self.scale_base.unsqueeze(dim=1) * base + self.scale_sp.unsqueeze(dim=1) * y
        
        y = self.mask.unsqueeze(1)[None,:, :] * y
        
        postacts = y.clone().reshape(dimension, batch, self.out_dim, self.in_dim)
        y = torch.sum(y.reshape(dimension,batch, self.out_dim, self.in_dim), dim=3).squeeze(1)  # shape (batch, out_dim)
        
        # y shape: (batch, out_dim); preacts shape: (batch, in_dim, out_dim)
        # postspline shape: (batch, in_dim, out_dim); postacts: (batch, in_dim, out_dim)
        # postspline is for extension; postacts is for visualization
        return y  # , preacts, postacts, postspline
    


    def update_grid_from_samples(self, x):
        '''
        update grid from samples

        Args:
        -----
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(model.grid.data)
        >>> x = torch.linspace(-3,3,steps=100)[:,None]
        >>> model.update_grid_from_samples(x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-3.0002, -1.7882, -0.5763,  0.6357,  1.8476,  3.0002]])
        '''
        
        
        
        batch = x.shape[0]
        # x = torch.einsum('dij,k->dikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(dimension, batch,  self.size).permute(0,2,1)
        
        x = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(batch,  self.size)
        
        x_pos = torch.sort(x, dim=1)[0]
        
        # x_pos = torch.sort(x, dim=0)[0]
        y_eval = torch.cat([coef2curve(x_pos[i].unsqueeze(1), self.grid[i], self.coef[i], self.k, device=self.device).unsqueeze(0) for i in range(batch)],dim=0)
        
        num_interval = self.grid.shape[2] - 1
        ids = [int(self.size / num_interval * i) for i in range(num_interval)] + [-1]
        
        
        # ids = [int(self.size / num_interval * i) for i in range(num_interval)] + [-1]
        # grid_adaptive = x_pos[:, ids, :]
        
        # Initialize grid_adaptive
        grid_adaptive = torch.zeros_like(self.grid)  # Shape: (embedding_dim, num_intervals, num_grid_points)
        
        x_pos = x_pos.unsqueeze(1)
        
        # Adaptive grid should be computed for each embedding dimension and grid interval
        for i in range(self.grid.shape[0]):  # Iterate over embedding dimensions (e.g., 10)
            grid_adaptive_embedding = x_pos[i,:,ids]  
            grid_adaptive[i, :, :] = grid_adaptive_embedding.unsqueeze(0) 
        # grid_adaptive = torch.cat(grid_adaptive, dim=0)  # Combine them into a tensor
        
        margin = 0.01
        grid_uniform = []
        for i in range(self.grid.shape[0]):
            grid_adaptive_emb = grid_adaptive[i]
            grid_uniform.append(torch.cat(
            [grid_adaptive_emb[:, [0]] - margin + (grid_adaptive_emb[:,  [-1]] - grid_adaptive_emb[:, [0]] + 2 * margin) * a for a in
            np.linspace(0, 1, num=self.grid.shape[2])], dim=1).unsqueeze(0))
        grid_uniform = torch.cat(grid_uniform, dim = 0)
        
        self.grid.data = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        
        # self.coef.data = curve2coef(x_pos, y_eval, self.grid, self.k, device=self.device)
        self.coef.data = torch.cat([curve2coef(x_pos[i].permute(1,0), y_eval[i], self.grid[i], self.k, device=self.device).unsqueeze(0) for i in range(batch)],dim=0)
        
        

    def initialize_grid_from_parent(self, parent, x):
        '''
        update grid from a parent KANLayer & samples

        Args:
        -----
            parent : KANLayer
                a parent KANLayer (whose grid is usually coarser than the current model)
            x : 2D torch.float
                inputs, shape (number of samples, input dimension)

        Returns:
        --------
            None

        Example
        -------
        >>> batch = 100
        >>> parent_model = KANLayer(in_dim=1, out_dim=1, num=5, k=3)
        >>> print(parent_model.grid.data)
        >>> model = KANLayer(in_dim=1, out_dim=1, num=10, k=3)
        >>> x = torch.normal(0,1,size=(batch, 1))
        >>> model.initialize_grid_from_parent(parent_model, x)
        >>> print(model.grid.data)
        tensor([[-1.0000, -0.6000, -0.2000,  0.2000,  0.6000,  1.0000]])
        tensor([[-1.0000, -0.8000, -0.6000, -0.4000, -0.2000,  0.0000,  0.2000,  0.4000,
          0.6000,  0.8000,  1.0000]])
        '''
        
        embedding = x.shape[0]
        # preacts: shape (batch, in_dim) => shape (size, batch) (size = out_dim * in_dim)
        x_eval = torch.einsum('ij,k->ikj', x, torch.ones(self.out_dim, ).to(self.device)).reshape(embedding,
                                                                                                  self.size).permute(1,
                                                                                                                     0)
        
        x_pos = parent.grid
        sp2 = KANLayer(in_dim=1, out_dim=self.size,k=1, num=x_pos.shape[1] - 1, scale_base=0., device=self.device)
        sp2.coef.data = curve2coef(sp2.grid, x_pos, sp2.grid, k=1, device=self.device)
        y_eval = coef2curve(x_eval, parent.grid, parent.coef, parent.k, device=self.device)
        
        percentile = torch.linspace(-1, 1, self.num + 1).to(self.device)
        self.grid.data = sp2(percentile.unsqueeze(dim=1)).permute(1, 0)
        self.coef.data = curve2coef(x_eval, y_eval, self.grid, self.k, self.device)
        


# MLP Layer

class MLPLayer(nn.Module):
    """
    Standard MLP Layer with linear transformation and activation.
    """
    def __init__(self, input_dim, output_dim):
        super(MLPLayer, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, device = 'cuda')
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# MoE 

class MoE(nn.Module):
    """
    Mixture of Experts Module.
    """
    def __init__(self, input_dim, output_dim, num_experts=4, hidden_dim=64, expert_type='mlp'):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            if expert_type == 'mlp':
                self.experts.append(MLPLayer(input_dim, hidden_dim))
            elif expert_type == 'kan':
                self.experts.append(KANLayer(in_dim=input_dim, out_dim=hidden_dim,num=input_dim*hidden_dim, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True
                 ))
            else:
                raise ValueError("Expert type must be 'mlp' or 'kan'.")

        # Gating network outputs probabilities over experts
        self.gating_network = nn.Sequential(
            nn.Linear(input_dim, num_experts, device = 'cuda'),
            nn.Softmax(dim=-1)
        )

        # Output layer to map from hidden_dim to output_dim
        self.output_layer = nn.Linear(hidden_dim, output_dim, device = 'cuda')

    def forward(self, x):
        x = x.permute(0,2,1)
        B, N, L = x.shape
        x = x.reshape(B*N, L)
        # Get gating weights
        gating_weights = self.gating_network(x)  # Shape: [batch_size, num_experts]
        # Get expert outputs
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(x)  # Shape: [batch_size, hidden_dim]
            expert_outputs.append(expert_output.unsqueeze(2))  # Shape: [batch_size, hidden_dim, 1]

        # Concatenate expert outputs: Shape [batch_size, hidden_dim, num_experts]
        expert_outputs = torch.cat(expert_outputs, dim=2)
        
        # Weight the expert outputs by gating weights
        # Expand gating_weights to match expert_outputs dimensions
        # gating_weights = gating_weights.unsqueeze(1)  # Shape: [batch_size, 1, num_experts]
        # mixture_output = torch.bmm(expert_outputs, gating_weights.transpose(1, 2)).squeeze(2)  # Shape: [batch_size, hidden_dim]


        mixture_output = torch.einsum("BLE,BL->BL", expert_outputs,gating_weights).reshape(B, N, -1 ).contiguous().permute(0,2,1)
        # Final output layer
        # output = self.output_layer(mixture_output)  # Shape: [batch_size, output_dim]
        return mixture_output, gating_weights.unsqueeze(1)

# Multi-Scale Season Mixing

class MultiScaleSeasonMixing(nn.Module):
    """
    Bottom-up mixing season pattern using KANLayer.
    """
    def __init__(self, configs: Model3Config):
        super(MultiScaleSeasonMixing, self).__init__()
        self.down_sampling_layers1 = nn.ModuleList()
        self.down_sampling_layers2 = nn.ModuleList()
        for i in range(configs.down_sampling_layers):
            input_dim = configs.seq_len // (configs.down_sampling_window ** i)
            output_dim = configs.seq_len // (configs.down_sampling_window ** (i + 1))
            self.down_sampling_layers1.append(
                    MLPLayer(input_dim=output_dim,output_dim=output_dim))
            self.down_sampling_layers2.append(
                KANLayer(in_dim=input_dim, out_dim=output_dim, d_model=configs.d_model,num=input_dim*output_dim, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True
                 ))
            

    def forward(self, season_list):
        out_high = season_list[0]  # Shape: [B, N, T_high]
        out_season_list = [out_high]
        
        for i in range(len(season_list) - 1):
            
            self.down_sampling_layers2[i].update_grid_from_samples(out_high.permute(0,2,1).squeeze(0))
            # self.down_sampling_layers2[i].initialize_grid_from_parent(self.down_sampling_layers2[i], out_high.permute(0,2,1).squeeze(0))
            
            out_low_res = self.down_sampling_layers2[i](out_high.permute(0,2,1).squeeze(0)).unsqueeze(0).permute(0,2,1)  # Maps over time dimension
            # out_low_res = self.down_sampling_layers1[i](out_low_res.permute(0,2,1)).permute(0,2,1)
            out_low = season_list[i + 1] + out_low_res
            out_high = out_low
            out_season_list.append(out_high)
            
        
        return out_season_list
    
    

# Multi-Scale Trend Mixing

class MultiScaleTrendMixing(nn.Module):
    """
    Top-down mixing trend pattern using KANLayer.
    """
    def __init__(self, configs: Model3Config):
        super(MultiScaleTrendMixing, self).__init__()
        self.up_sampling_layers1 = nn.ModuleList()
        self.up_sampling_layers2 = nn.ModuleList()
        
        for i in reversed(range(configs.down_sampling_layers)):
            input_dim = configs.seq_len // (configs.down_sampling_window ** (i + 1))
            output_dim = configs.seq_len // (configs.down_sampling_window ** i)
            self.up_sampling_layers1.append(
                    MLPLayer(input_dim = input_dim,output_dim=output_dim))
            self.up_sampling_layers2.append(
                KANLayer(in_dim=input_dim, out_dim=output_dim, d_model=configs.d_model,num=input_dim*output_dim, k=3, noise_scale=0.1, scale_base=1.0, scale_sp=1.0,
                 base_fun=torch.nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True
                 ))
            


    def forward(self, trend_list):
        # Mixing low->high
        trend_list_reverse = trend_list[::-1]
        out_low = trend_list_reverse[0]
        out_trend_list = [out_low]

        for i in range(len(trend_list_reverse) - 1):
            self.up_sampling_layers2[i].update_grid_from_samples(out_low.permute(0,2,1).squeeze(0))
            # self.up_sampling_layers2[i].initialize_grid_from_parent(self.down_sampling_layers2[i], out_low.permute(0,2,1).squeeze(0))
            out_high_res = self.up_sampling_layers2[i](out_low.permute(0,2,1).squeeze(0)).unsqueeze(0).permute(0,2,1)
            # out_high_res = self.up_sampling_layers1[i](out_high_res.permute(0,2,1)).permute(0,2,1)
            out_high = trend_list_reverse[i + 1] + out_high_res
            out_low = out_high
            out_trend_list.append(out_low)
            
        out_trend_list.reverse()
        
        
        return out_trend_list

#  Past Decomposable Mixing 

class PastDecomposableMixing(nn.Module):
    """
    Past Decomposable Mixing block with MoE integration.
    """
    def __init__(self, configs: Model3Config):
        super(PastDecomposableMixing, self).__init__()
        self.seq_len = configs.seq_len
        self.channel_independence = configs.channel_independence
        self.d_model = configs.d_model

        # Series decomposition
        self.decomposition = series_decomp(configs.moving_avg)
        
        if not self.channel_independence:
            self.cross_layer=[]
            for i in range(configs.down_sampling_layers + 1):
                
                input_dim = configs.seq_len // (configs.down_sampling_window ** i)
                self.cross_layer.append(
                        MLPLayer(input_dim=input_dim,output_dim=input_dim))
            

        # Mixing season and trend
        self.mixing_season = MultiScaleSeasonMixing(configs)
        self.mixing_trend = MultiScaleTrendMixing(configs)
        
        if not self.channel_independence:
            self.out_cross_layer = []
            for i in range(configs.down_sampling_layers + 1):

                input_dim = configs.seq_len // (configs.down_sampling_window ** i)
                # output_dim = configs.seq_len // (configs.down_sampling_window ** (i + 1))
                
                self.out_cross_layer.append(
                            MLPLayer(input_dim=input_dim,output_dim=input_dim))
            
            # self.out_cross_layer = MLPLayer(configs.d_model, configs.d_model)

        '''
        self.moe_layer = []
        # for i in range(configs.down_sampling_layers + 1):
                
        # input_dim = configs.seq_len // (configs.down_sampling_window ** i)
        self.moe_layer.append(input_dim=input_dim, 
                                output_dim=input_dim, 
                                num_experts=configs.num_experts, 
                                hidden_dim=configs.d_model, 
                                expert_type=configs.expert_type)
         '''

    def forward(self, x_list):
        # Decompose to obtain the season and trend
        season_list = []
        trend_list = []
        length_list = []
        
        for i,x in zip(range(len(x_list)), x_list):
            
            _, T,_ = x.size()
            length_list.append(T)
            season, trend = self.decomposition(x)
            
            if not self.channel_independence:
                season = self.cross_layer[i](season.permute(0,2,1)).permute(0,2,1)
                trend = self.cross_layer[i](trend.permute(0,2,1)).permute(0,2,1)

            
            season_list.append(season)
            trend_list.append(trend)
            
        # Bottom-up season mixing
        out_season_list = self.mixing_season(season_list)
        # Top-down trend mixing
        out_trend_list = self.mixing_trend(trend_list)

        out_list = []
        gating_weights_list = []
        for i, ori, out_season, out_trend, length in zip(range(len(x_list)), x_list, out_season_list, out_trend_list, length_list):
            # Combine season and trend
            out = out_season + out_trend
            if not self.channel_independence:
                out = ori + self.out_cross_layer[i](out.permute(0,2,1)).permute(0,2,1)
            
            # Apply MoE layer
            B, T, N = out.size()
            # out_flat = out.reshape(B * T, N)
            '''
            out_moe, gating_weights = self.moe_layer[i](out)
            print(out_moe.shape)
            '''# out = out_moe.reshape(B, T, N)
            out_list.append(out)
            # gating_weights_list.append(gating_weights)
        return out_list #, gating_weights_list

class HybridModel(nn.Module):
    
    def __init__(self, configs: Model3Config):
        super(HybridModel, self).__init__()
        self.configs = configs
        self.layer = configs.e_layers
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        self.d_model = configs.d_model

        # Series decomposition
        self.preprocess = series_decomp(configs.moving_avg)

        # Embedding layer
        self.enc_embedding = DataEmbedding(c_in=self.configs.num_channels, d_model=self.configs.d_model)

        # Past Decomposable Mixing layers
        self.pdm_blocks = nn.ModuleList([
            PastDecomposableMixing(configs) for _ in range(configs.e_layers)
        ])

        
        self.padding = 1 if torch.__version__ >= '1.5.0' else 2
        
        self.down_pool_conv = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=self.padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        
        self.down_pool_avg = torch.nn.AvgPool1d(self.configs.down_sampling_window)

        self.down_pool_max = torch.nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)




        # Normalize layers
        self.normalize_layers = nn.ModuleList([
             Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)

            # nn.LayerNorm(configs['d_model']) for _ in range(configs['down_sampling_layers'] + 1)
        ])

        # Prediction layers
        self.predict_layers = nn.ModuleList([
            nn.Linear(configs.seq_len // (configs.down_sampling_window ** i), configs.pred_len)
            for i in range(configs.down_sampling_window + 1)
        ])

        # Projection layer
        if self.channel_independence:
            self.projection_layer = nn.Linear(configs.d_model, 1)
        else:
            self.projection_layer = nn.Linear(configs.d_model, configs.c_out)
            
    def pre_enc(self, x_list):
        if self.channel_independence:
            return x_list, None
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return out1_list, out2_list

    def multi_scale_process_inputs(self, x_enc):
        x_enc_list = [x_enc]
        
        if self.configs.down_sampling_method == 'max':
            down_pool = self.down_pool_max
        elif self.configs.down_sampling_method == 'avg':
            down_pool = self.down_pool_avg
        elif self.configs.down_sampling_method == 'conv':
            
            down_pool = self.down_pool_conv

        for i in range(self.configs.down_sampling_layers):
            x_enc_down = down_pool(x_enc.permute(0, 2, 1)).permute(0, 2, 1)
            x_enc_list.append(x_enc_down)
            x_enc = x_enc_down
        return x_enc_list

    def future_multi_mixing(self, B, enc_out_list):
        dec_out_list = []
        for i, enc_out in enumerate(enc_out_list):
            dec_out = self.predict_layers[i](enc_out.permute(0, 2, 1)).permute(0, 2, 1)
            dec_out = self.projection_layer(dec_out)
            dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
            dec_out_list.append(dec_out)
        return dec_out_list

    def forecast(self, x_enc):
        x_enc_list = self.multi_scale_process_inputs(x_enc)
        
        x_list = []
        for i, x in zip(range(len(x_enc_list)), x_enc_list):
            x = self.normalize_layers[i](x, 'norm')
            x_list.append(x)

        enc_out_list=[]
        # Pre-encoding processing
        x_list = self.pre_enc(x_list)[0]
        
        for i, x in zip(range(len(x_list)), x_list):
            
            # enc_out = self.enc_embedding(x)
            # B, N, T  eg. 1, N, 1 
            enc_out_list.append(x)
            # downsample, B, N, T
        
        total_entropy = 0
        # Past Decomposable Mixing with MoE
        for i in range(self.layer):
            enc_out_list  = self.pdm_blocks[i](enc_out_list)
            
            '''for gating_weights in gating_weights_list:
                entropy = -torch.sum(gating_weights*torch.log(gating_weights + 1e-8), dim = 1).mean()
                total_entropy += entropy'''
        
        B = x_enc.size(0)
        # Future Multi Mixing
        dec_out_list = self.future_multi_mixing(B, enc_out_list=enc_out_list)
        
        
        dec_out = torch.stack(dec_out_list, dim=-1).sum(-1)
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out#, total_entropy

    def forward(self, x_enc):
        dec_out = self.forecast(x_enc)
        return dec_out#, total_entropy