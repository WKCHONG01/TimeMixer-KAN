import torch
import math
from typing import Optional, Tuple
import torch.nn as nn

class layerNorm(nn.Module):
    def __init__(self, d_model):
        super(layerNorm, self).__init__()
        self.layernorm = nn.LayerNorm(d_model)
    def forward(self, x):
        norm = self.layernorm(x)
        res = norm - torch.mean(norm, dim= 1).unsqueeze(1).repeat(1,x.shape[1],1)
        return res

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride): # 7 , 1
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # 32x 92 x 7
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1) # # 32x 3 x 7 (first)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1) # 32x 3 x 7 (last)
        x = torch.cat([front, x, end], dim=1) # 32 x 98 x 7
        x = self.avg(x.permute(0, 2, 1)) # 32 x 7 x 92 
        x = x.permute(0, 2, 1) # 32 x 92 x 7
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # 32x 92 x 7
        moving_mean = self.moving_avg(x) 
        # 32 x 92 x 7
        res = x - moving_mean
        # 32 x 92 x 7
        return res, moving_mean # 32 x 92 x 7 (residuals),  32 x 92 x 7 (mean)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(DataEmbedding, self).__init__()
        self.token_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
    def forward(self,x):
        x = self.token_embedding(x.permute(0,2,1)).transpose(2,1) + self.position_embedding(x)
        return x
    
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels = c_in, out_channels = d_model, kernel_size = 1,  bias = False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')


    def forward(self, x):
        x = self.tokenConv(x)
        return x

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len,d_model).float()
        pe.requires_grad = False
        
        position = torch.arange(0,max_len).float().unsqueeze(1)
        div_term = (torch.arange(0,d_model,2).float() * -(math.log(10000.0)/d_model)).exp()
        
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
        
        
    def forward(self, x):
        _, seq_len, _ = x.size()
        position = self.pe[:,:seq_len]
        return position



class Autocorrelation(nn.Module):
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, training = False):
        super(Autocorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.training = training

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the training phase.
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1) # 32, 96 
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1] # 1 x 25 
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg
    
    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg


    def forward(self, query, key, value, attn_mask):
        B, L, H, E = query.shape
        _, S, _, D = value.shape
        
        if L > S:
            zeros = torch.zeros_like(query[:, :(L - S), :]).float()
            value = torch.cat([value, zeros], dim=1)
            key = torch.cat([key, zeros], dim=1)
        else:
            value = value[:, :L, :, :]
            key = key[:, :L, :, :]
        q_fft = torch.fft.rfft(query.permute(0,2,3,1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(key.permute(0,2,3,1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, n=L, dim =-1)
        if self.training:
            V = self.time_delay_agg_training(value.permute(0,2,3,1).contiguous(), corr).permute(0,3,1,2)
        else:
            V = self.time_delay_agg_inference(value.permute(0,2,3,1).contiguous(), corr).permute(0,3,1,2)
        
        return V.contiguous(), corr.permute(0,3,1,2)

class AutoCorrelationLayer(nn.Module):
    def __init__(self, inner_attention, num_heads, d_models, d_keys, d_values):
        super().__init__()
        self.inner_attention = inner_attention
        self.d_models = d_models
        self.num_heads = num_heads
        self.d_keys = d_keys
        self.d_values = d_values

        self.k_proj = nn.Linear(in_features=self.d_models, out_features=d_keys*num_heads, bias = False)
        self.q_proj = nn.Linear(in_features=d_models, out_features=d_keys*num_heads, bias = False)
        self.v_proj = nn.Linear(in_features=d_models, out_features=d_values*num_heads, bias = False)
        self.o_proj = nn.Linear(in_features=d_values*num_heads, out_features=d_models, bias = False )



    def forward(self, query, key, value, attn_mask):
        B, L, embed_dim = query.shape
        _, S, _ = key.shape
        H = self.num_heads
        
        queries = self.q_proj(query).view(B, L, H, -1)
        keys = self.k_proj(key).view(B, S, H, -1)
        values = self.v_proj(value).view(B, S, H, -1)
        
        out, attn = self.inner_attention(
            queries,
            keys, 
            values, 
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.o_proj(out), attn

class ModelConfig:
    def __init__(
        self,
        hidden_size = 512,
        num_hidden_layers = 12,
        num_attention_heads = 2,
        num_channels = 1,
        moving_avg = 3,
        activation = "relu",
        e_layers = 3,
        d_layers = 2,
        d_keys = 1024,
        d_values = 1024,
        factor = 1,
        dropout = 0.1,
        training = True,
        d_ff = 1024,
        **kwargs

    ):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.moving_avg = moving_avg
        self.activation = activation
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_keys = d_keys
        self.d_values = d_values
        self.factor = factor
        self.dropout = dropout
        self.training = training
        self.d_ff = d_ff

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, moving_avg = 25, activation = "relu", dropout= 0.1, d_ff=None,) :
        super().__init__()
        self.d_ff = d_ff or 4*d_model
        self.attention = attention
        self.d_model = d_model
        self.moving_avg = moving_avg,
        self.activation = nn.functional.relu if activation == "relu" else nn.functional.gelu
        self.dropout = nn.Dropout(dropout)

        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.mlp1 = nn.Linear(in_features=self.d_model, out_features=self.d_ff)
        self.mlp2 = nn.Linear(in_features=self.d_ff, out_features=self.d_model)
        
        #Autoformer
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        


    def forward(self, x, attn_mask = None):
        res = x
        x, attn = self.attention(
        x,x,x, attn_mask
        )
        
        x = res + self.dropout(x)
        res = x
        x, _ = self.decomp1(x)
        # x = self.dropout(self.activation(self.mlp1(x)))
        # x = self.dropout(self.mlp2(x))
        x = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        x = self.dropout(self.conv2(x).transpose(-1, 1))
        
        
        seasonal, trend = self.decomp2(x + res)
        return seasonal, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer = None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask = None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        
        return x, attns

class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                     padding_mode = "circular", bias=False)
        self.activation = nn.functional.relu if activation == "relu" else nn.functional.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])
        x, trend1 = self.decomp1(x)
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        return x, residual_trend


class Decoder(nn.Module):
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend




class Model(nn.Module):
    def __init__(self, config: ModelConfig ):  
        super().__init__() 
        self.config = config
        self.num_channels = config.num_channels
        self.hidden_size = config.hidden_size
        self.kernel_size = config.moving_avg
        self.d_ff = config.d_ff
        self.activation = config.activation
        self.e_layers = config.e_layers
        self.d_layers = config.d_layers
        self.embedding = DataEmbedding(self.num_channels, self.hidden_size)
        self.dropout = config.dropout
        self.series_decomp = series_decomp(kernel_size=self.kernel_size)

        self.num_heads = config.num_attention_heads
        self.d_keys = config.d_keys
        self.d_values = config.d_values

        self.training = config.training
        
        self.encoder = Encoder(
            [
                EncoderLayer(
                    attention = AutoCorrelationLayer(
                        inner_attention = Autocorrelation(
                            False, 
                            config.factor, 
                            attention_dropout=config.dropout,
                            training=self.training

                        ),
                        num_heads = self.num_heads,
                        d_models = self.hidden_size,
                        d_keys = self.d_keys,
                        d_values = self.d_values,
                    ),
                    
                    d_model=self.hidden_size,
                    moving_avg=3,
                    activation=self.activation,
                    d_ff=self.d_ff
                ) for l in range(self.e_layers)
            ],
            norm_layer= layerNorm(d_model=self.hidden_size)
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        Autocorrelation(True, config.factor, attention_dropout=self.dropout, training= self.training),
                        d_models=self.hidden_size, num_heads=self.num_heads, d_keys=self.d_keys, d_values = self.d_values),
                    AutoCorrelationLayer(
                        Autocorrelation(False, config.factor, attention_dropout=self.dropout, training= self.training),
                        d_models=self.hidden_size, num_heads=self.num_heads, d_keys=self.d_keys, d_values = self.d_values),
                    self.hidden_size, # 512
                    self.num_channels, # 
                    self.d_ff, #  
                    moving_avg=self.kernel_size,
                    dropout=self.dropout,
                    activation=self.activation,
                )
                for l in range(config.d_layers)
            ],
            norm_layer=layerNorm(self.hidden_size),
            projection=nn.Linear(self.hidden_size, self.num_channels, bias=True)
        )

    def forward(self, x: torch.FloatTensor, ground_truth: torch.FloatTensor) -> torch.Tensor:
        embed_x = self.embedding(x)
        seasonal, trend = self.series_decomp(x)
        mean = torch.mean(x, dim = 1).unsqueeze(1).repeat(1, ground_truth.shape[1], 1)
        zeros = torch.zeros(ground_truth.shape, device = x.device)
        
        trend_dec = torch.cat([trend, mean], dim = 1)
        seasonal_dec = torch.cat([seasonal, zeros], dim = 1)
        
        embed_dec = self.embedding(seasonal_dec)

        # using same mean for future? but future might have different mean of distribution
        enc_out, attn = self.encoder(embed_x)

        seasonal, trend = self.decoder(embed_dec, enc_out, trend = trend_dec)

        dec_out=seasonal + trend

        return dec_out[:, -ground_truth.shape[1]:,:]