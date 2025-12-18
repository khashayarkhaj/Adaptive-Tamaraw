import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math
from .common_layer import EncoderLayer ,MultiHeadAttention ,Conv ,PositionwiseFeedForward ,LayerNorm
from utils.file_operations import predict_model_size
# I broke down the original model into modules so that pretraining and dual learning would be easier
import torch.nn as nn

class EmbeddingModule(nn.Module):
    def __init__(self, num_vocab, embedding_size):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, embedding_size, padding_idx=0)
    
    def forward(self, x):
        return self.emb(x.view(x.size(0), -1))

class TransformerModule(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, 
                 total_key_depth, total_value_depth, filter_size, lens,
                 max_length=5000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0):
        super().__init__()
        self.transformer_enc = Encoder(
            embedding_size, hidden_size, num_layers, num_heads,
            total_key_depth, total_value_depth, filter_size,
            lens=lens, max_length=lens, input_dropout=input_dropout,
            window=500, layer_dropout=layer_dropout,
            attention_dropout=attention_dropout, relu_dropout=relu_dropout,
            use_mask=False, verbose=False
        )
        self.ln = LayerNorm(hidden_size)
    
    def forward(self, x):
        x = self.transformer_enc(x)
        x = self.ln(x)
        return x.permute(0, 2, 1).contiguous()

class CompressionModule(nn.Module):
    def __init__(self, lens, hidden_size, compression_size=2048):
        super().__init__()
        self.fc1 = nn.Linear(lens, compression_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.9)
        self.fc2 = nn.Linear(compression_size, 1)
        self.bn = nn.BatchNorm1d(hidden_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn(x)
        return x.view(x.shape[0], x.shape[1])

class ClassifierModule(nn.Module): # supports dual outputs as well
    def __init__(self, input_size, hidden_size=2048, num_classes=100, num_classes_2 = None):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=0.9)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.num_classes2 = num_classes_2
        self.multi_task_enabled = False
        if num_classes_2 is not None:
            self.multi_task_enabled = True
            self.task2_fc2 = nn.Linear(hidden_size, num_classes_2)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        if self.multi_task_enabled:
            task1_output = self.fc2(x)
            task2_output = self.task2_fc2(x)
            
            return task1_output, task2_output
        else:
            x = self.fc2(x)
            return x

class UTransformer(nn.Module):
    def __init__(self, num_vocab, embedding_size, hidden_size, num_layers, 
                 num_heads, total_key_depth, total_value_depth, filter_size, 
                 lens, max_length=5000, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, classes=100,
                 compression_size=2048, classifier_hidden_size=2048, 
                 verbose=False, num_classes2 = None):
        super().__init__()
        
        # Initialize all modules
        self.embedding = EmbeddingModule(num_vocab, embedding_size)
        
        self.transformer = TransformerModule(
            embedding_size, hidden_size, num_layers, num_heads,
            total_key_depth, total_value_depth, filter_size, lens,
            max_length, input_dropout, layer_dropout,
            attention_dropout, relu_dropout
        )
        
        self.compression = CompressionModule(
            lens, hidden_size, compression_size
        )
        
        self.classifier = ClassifierModule(
            input_size= hidden_size, hidden_size = classifier_hidden_size, num_classes= classes,
            num_classes_2= num_classes2
        )
        
        self.verbose = verbose
    
    def forward(self, story):
        # Embedding
        x = self.embedding(story)
        if self.verbose:
            print("Shape after embedding:", x.shape)
        
        # Transformer processing
        x = self.transformer(x)
        if self.verbose:
            print("Shape after transformer:", x.shape)
        
        # Compression
        x = self.compression(x)
        if self.verbose:
            print("Shape after compression:", x.shape)
        
        # Classification
        if not self.classifier.multi_task_enabled:
            x = self.classifier(x)
            if self.verbose:
                print("Shape after classifier:", x.shape)
            
            return x
        else:
            task1_output, task2_output = self.classifier(x)
            if self.verbose:
                print("Shape of task1 output after classifier:", task1_output.shape)

                print("Shape of task2 output after classifier:", task2_output.shape)
            return task1_output, task2_output


    def enable_multi_task(self):
        self.classifier.multi_task_enabled = True
    def disable_multi_task(self):
        self.classifier.multi_task_enabled = False

        

class Encoder(nn.Module):
    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size,lens, window=None,max_length=100, input_dropout=0.0, layer_dropout=0.0, 
                 attention_dropout=0.0, relu_dropout=0.0, use_mask=False, act=False, verbose = False):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """
        
        super(Encoder, self).__init__()
        
        #self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        ## for t
        #self.position_signal = _gen_timing_signal(num_layers, hidden_size)

        self.num_layers = num_layers

        params =(hidden_size, 
                 total_key_depth or hidden_size,
                 total_value_depth or hidden_size,
                 filter_size, 
                 num_heads, 
                 lens,
                 window,
                 _gen_bias_mask(max_length) if use_mask else None,
                 layer_dropout, 
                 attention_dropout, 
                 relu_dropout)
        
        

        self.proj_flag = False
        self.embedding_proj = nn.Linear(embedding_size, hidden_size, bias=False)

        self.enc = EncoderLayer(*params)
        
        self.enc.verbose = verbose
        
        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        self.verbose = verbose

    def forward(self, inputs):

        #Add input dropout
        x = self.input_dropout(inputs)

        x = self.embedding_proj(x)
        if self.verbose:
            print("Shape after transformer embedding projection:", x.shape)
        for l in range(self.num_layers):
            #x += self.timing_signal[:, :inputs.shape[1], :].type_as(inputs.data)
            #x += self.position_signal[:, l, :].unsqueeze(1).repeat(1,inputs.shape[1],1).type_as(inputs.data)
            x = self.enc(x)
        return x

if __name__ == '__main__':
    classes=95
    lens = 5000  # sequence length

# Initialize your model
    model = UTransformer(
            num_vocab=3,
            embedding_size=128,
            hidden_size=1024,
            num_layers=1,
            num_heads=1,
            total_key_depth=512,
            total_value_depth=512,
            filter_size=512,
            classes=classes,
            lens=lens,
            input_dropout=0.0,
            layer_dropout=0.0,
            attention_dropout=0.1,
            relu_dropout=0.1,
            verbose= True,
        )
    
    

    # model = UTransformer(
    #     num_vocab=3,
    #     embedding_size=32,
    #     hidden_size=256,
    #     num_layers=1,
    #     num_heads=1,
    #     total_key_depth=64,
    #     total_value_depth=64,
    #     filter_size=64,
    #     classes=classes,
    #     lens=lens,
    #     input_dropout=0.0,
    #     layer_dropout=0.0,
    #     attention_dropout=0.1,
    #     relu_dropout=0.1,
    #     verbose= True,
    #     first_linear_layer= 128,
    #     second_linear_layer= 128
    # )

    # Generate a random sequence
    sequence_length = 5000
    data = torch.randint(low=-1, high=2, size=(sequence_length,))  # Generates values -1, 0, 1
    data[data == -1] = 0
    data = data.unsqueeze(0)  # Add batch dimension

    # Test the model
    model.eval()
    with torch.no_grad():
        data = data.view([-1,lens,1]).long()
        output = model(data)
        print("Output shape:", output.shape)
        # print("Output:", output)
    
    print('estimated model size is ', predict_model_size(model = model))
# python3 -m models.WF_Transformer.UTransformer