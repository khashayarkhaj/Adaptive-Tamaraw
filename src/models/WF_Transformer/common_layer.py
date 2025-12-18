import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
from .diagonaled_mm_tvm import diagonaled_mm as diagonaled_mm_tvm, mask_invalid_locations
from .sliding_chunks import sliding_chunks_matmul_qk, sliding_chunks_matmul_pv
from .sliding_chunks import sliding_chunks_no_overlap_matmul_qk, sliding_chunks_no_overlap_matmul_pv
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,lens,window,
                 bias_mask=None, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0, verbose = False):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output (Not in the function!)
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """
        
        super(EncoderLayer, self).__init__()
        
        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth, 
                                                       hidden_size, num_heads, lens,window,bias_mask, attention_dropout)
        
        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 lens,layer_config='cccc', 
                                                                 padding = 'both', dropout=relu_dropout)

        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

        self.verbose = verbose
        
    def forward(self, inputs):
        x = inputs
        if self.verbose:
            print("Shape of input to tranformer encoder :", x.shape)
        # Layer Normalization
        x_norm = self.layer_norm_mha(x)

        # Multi-head attention
        y = self.multi_head_attention(x_norm, x_norm, x_norm)
        
        # Dropout and residual
        x = self.dropout(x+y)
        if self.verbose:
            print("Shape after mha of tranformer encoder :", x.shape)
        
        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual
        y = self.dropout(x+y)

        if self.verbose:
            print("Shape after pff of tranformer encoder :", y.shape)
        return y



class MultiHeadAttention(nn.Module):
    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth, 
                 num_heads,lens,window=None, bias_mask=None, dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(MultiHeadAttention, self).__init__()
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))
            
        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5
        self.bias_mask = bias_mask
        self.window = window
        self.lens = lens
        # Key and query depth will be same
        self.query_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.key_linear = nn.Linear(input_depth, total_key_depth, bias=False)
        self.value_linear = nn.Linear(input_depth, total_value_depth, bias=False)
        self.output_linear = nn.Linear(total_value_depth, output_depth, bias=False)
        self.qkv_conv = SepConv2d(1, 3, kernel_size=(9,1), stride=1, padding=(4,0), dilation=1)
        self.dropout = nn.Dropout(dropout)
        coord1 = -torch.arange(lens)
        coord2 = torch.arange(lens)
        coords = torch.stack(torch.meshgrid([coord1, coord2]))
        coords = coords.permute(1, 2, 0).contiguous()
        self.relative_position_index = coords.sum(-1)+lens
        self.relative_position_bias_table = nn.Parameter(torch.zeros([lens*2]))  # [2*Mh-1 * 2*Mw-1, nH]
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.relu = nn.GELU()
    
    def _split_heads(self, x):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        """
        if len(x.shape) != 3:
            raise ValueError("x must have rank 3")
        shape = x.shape
        return x.view(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3)
    
    def _merge_heads(self, x):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth]
        """
        if len(x.shape) != 4:
            raise ValueError("x must have rank 4")
        shape = x.shape
        return x.permute(0, 2, 1, 3).contiguous().view(shape[0], shape[2], shape[3]*self.num_heads)
        
    def forward(self, queries, keys, values, src_mask=None):
        
        x=queries.unsqueeze(1)
        x=self.qkv_conv(x)
        x=x.permute(1, 0, 2, 3)
        x=self.relu(x)
        queries = x[0]
        keys = x[1]
        values = x[2]
        # Do a linear for each component
        queries = self.query_linear(queries)
        keys = self.key_linear(keys)
        values = self.value_linear(values)
        
        # Split into multiple heads
        queries = self._split_heads(queries)
        keys = self._split_heads(keys)
        values = self._split_heads(values)
        
        # Scale queries
        queries *= self.query_scale
        
        # Combine queries and keys
        if self.window is not None:
            attn_weights = sliding_chunks_matmul_qk(queries.permute(0, 2, 1, 3), keys.permute(0, 2, 1, 3), self.window, padding_value=0)
            #mask_invalid_locations(attn_weights, self.window, 1, False)
            relative_position_index = torch.arange(self.window*2+1).repeat(self.lens,1)
            relative_position_bias = self.relative_position_bias_table[relative_position_index.view(-1)].view([attn_weights.shape[1],attn_weights.shape[3]])
            attn_weights = attn_weights.permute(0, 2, 1, 3)+relative_position_bias
            attn_weights = attn_weights.permute(0, 2, 1, 3)
            attn_weights_float = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
            attn_weights = attn_weights_float.type_as(attn_weights)
            attn_probs = self.dropout(attn_weights_float.type_as(attn_weights))
            values = values.permute(0, 2, 1, 3)
            attn = 0
            attn += sliding_chunks_matmul_pv(attn_probs, values, self.window)
            contexts = attn.permute(0, 2, 1, 3)
        else:
            logits = torch.matmul(queries, keys.permute(0, 1, 3, 2))
            relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view([logits.shape[2],logits.shape[3]]).unsqueeze(0).unsqueeze(0)
            logits = logits+relative_position_bias

            if src_mask is not None:
                logits = logits.masked_fill(src_mask, -np.inf)
            
            # Add bias to mask future values
            if self.bias_mask is not None:
                logits += self.bias_mask[:, :, :logits.shape[-2], :logits.shape[-1]].type_as(logits.data)
        
            # Convert to probabilites
            weights = nn.functional.softmax(logits, dim=-1)
        
            # Dropout
            weights = self.dropout(weights)
            # Combine with values to get context
            contexts = torch.matmul(weights, values)
        
        # Merge heads
        contexts = self._merge_heads(contexts)
        #contexts = torch.tanh(contexts)
        
        # Linear to get output
        outputs = self.output_linear(contexts)
        
        return outputs

class SepConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super(SepConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels)
        self.relu = nn.GELU()
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.relu(x)
        x = self.pointwise(x)
        return x

class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """
    def __init__(self, input_size, output_size, kernel_size, pad_type, dilation=1):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data), 
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        kernel_size1 = dilation*(kernel_size-1)+1
        padding = (kernel_size1 - 1,0)
        #if pad_type == 'left' else (kernel_size1//2, (kernel_size1 - 1)//2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv2d(input_size, output_size, kernel_size=(kernel_size,1), stride=1, padding=0, dilation=dilation)

    def forward(self, inputs):
        x = self.pad(inputs.permute(0, 2, 1))
        x = x.unsqueeze(-1)
        outputs = self.conv(x).squeeze(-1).permute(0, 2, 1)
        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """
    def __init__(self, input_depth, filter_size, output_depth, lens ,layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data), 
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()
        
        layers = []
        sizes = ([(input_depth, filter_size)] + 
                 [(filter_size, filter_size)]*(len(layer_config)-2) + 
                 [(filter_size, output_depth)])
        d = 1
        i = 1
        dilation_rate = int(((lens-1)/8)**0.33)
        if abs(lens-1-8*(dilation_rate**3+dilation_rate**2+dilation_rate)) >= abs(lens-8*((dilation_rate+1)**3+(dilation_rate+1)**2+(dilation_rate+1))):
            dilation_rate = dilation_rate + 1
        if dilation_rate > 9:
            dilation_rate = 9
        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=9, pad_type=padding, dilation=d))
                d = d*dilation_rate
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.GELU()#nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)

        return x


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-9):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class LabelSmoothing(nn.Module):
    """NLL loss with label smoothing. """ 
    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module. :param smoothing: label smoothing factor"""
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class StepOpt:
    "Optim wrapper that implements rate."
    def __init__(self,datasize,batchsize,optimizer, initial_learning_rate = 0.0003, decay_rate = 0.1):
        self.optimizer = optimizer
        self.epochsize = math.ceil(datasize/batchsize)
        self._step = 0
        self._rate = 0
        self.initial_learning_rate = initial_learning_rate
        self.decay_rate = decay_rate
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self):
        "Implement `lrate` above"
        if self._step <= self.epochsize*20:
            return self.initial_learning_rate
        elif self._step <= self.epochsize*30:
            return self.initial_learning_rate * self.decay_rate
        else:
            return self.initial_learning_rate * (self.decay_rate ** 2)
'''self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))'''
