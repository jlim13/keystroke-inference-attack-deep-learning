import torch.nn as nn
import torch
import random


class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim,
            dropout, device, max_length = 300):

        super().__init__()
        self.device = device

        self.tok_embedding = nn.Linear(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([EncoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.LeakyReLU( inplace=False)

        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad = False)

    def forward(self, src, src_mask):

        #src = [batch size, src len]
        #src_mask = [batch size, src len]
        batch_size = src.shape[0]
        src_len = src.shape[1]

        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = torch.nn.Parameter(torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1), requires_grad = False)


        # pos = [batch size, src len]
        # print (self.tok_embedding(src).shape) # torch.Size([1, 361, 128])
        # print (self.scale)
        # print (self.pos_embedding(pos).shape) # torch.Size([1, 361, 128])
        # print ((self.tok_embedding(src) * self.scale).shape)

        agg = (self.relu(self.tok_embedding(src)) * self.scale) + self.pos_embedding(pos)

        # print (agg.shape)
        src = self.dropout(agg )

        # print (pos)
        #src = [batch size, src len, hid dim]

        for layer in self.layers:
            src = layer(src, src_mask)

        #src = [batch size, src len, hid dim]

        return src

class EncoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):

        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):

        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, src len]
        #self attention
        _src, _ = self.self_attention(src, src, src, src_mask)

        #dropout, residual connection and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        #positionwise feedforward
        _src = self.positionwise_feedforward(src)

        #dropout, residual and layer norm
        src = self.layer_norm(src + self.dropout(_src))

        #src = [batch size, src len, hid dim]

        return src

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()

        assert hid_dim % n_heads == 0

        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads

        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)

        self.fc_o = nn.Linear(hid_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

        # self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([self.head_dim])), requires_grad = False)

        # self.scale.requires_grad = False

    def forward(self, query, key, value, mask = None):

        batch_size = query.shape[0]

        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]

        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)

        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]

        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale

        #energy = [batch size, n heads, query len, key len]

        if mask is not None:

            energy = energy.masked_fill(mask == 0, -1e10)

        attention = torch.softmax(energy, dim = -1)

        #attention = [batch size, n heads, query len, key len]

        x = torch.matmul(self.dropout(attention), V)

        #x = [batch size, n heads, query len, head dim]

        x = x.permute(0, 2, 1, 3).contiguous()

        #x = [batch size, query len, n heads, head dim]

        x = x.view(batch_size, -1, self.hid_dim)

        #x = [batch size, query len, hid dim]

        x = self.fc_o(x)

        #x = [batch size, query len, hid dim]

        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()

        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        #x = [batch size, seq len, hid dim]

        x = self.dropout(torch.relu(self.fc_1(x)))

        #x = [batch size, seq len, pf dim]

        x = self.fc_2(x)

        #x = [batch size, seq len, hid dim]

        return x

# class Decoder_BN(nn.Module):
#
#     def __init__(self, decoder):
#
#         super().__init__()
#         self.decoder = decoder
#         # self.bn =
#
#     def forward(self, input):



class Decoder(nn.Module):

    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim,
            dropout, device, max_length = 70):

        super().__init__()
        self.device = device

        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)

        self.layers = nn.ModuleList([DecoderLayer(hid_dim,
                                                  n_heads,
                                                  pf_dim,
                                                  dropout,
                                                  device)
                                     for _ in range(n_layers)])

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

        # self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        self.scale = torch.nn.Parameter(torch.sqrt(torch.FloatTensor([hid_dim])), requires_grad = False)
        # self.scale.requires_grad = False
        self.src_pad_idx = -1
        self.trg_pad_idx = 1

    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        src_len = src.shape[1]
        batch_size = src.shape[0]
        src_mask = torch.ones([batch_size, 1 , 1, src_len], dtype = torch.bool, device = self.device)
        # src_mask = torch.nn.Parameter( torch.ones([batch_size, 1 , 1, src_len], dtype = torch.bool ), requires_grad = False)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        #trg_pad_mask = [batch size, 1, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()

        # trg_sub_mask = torch.nn.Parameter(torch.tril(torch.ones((trg_len, trg_len))).bool(), requires_grad = False)

        #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, trg, enc_src):

        src_mask = self.make_src_mask(enc_src)
        trg_mask = self.make_trg_mask(trg)

        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]

        batch_size = trg.shape[0]
        trg_len = trg.shape[1]

        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        # pos = torch.nn.Parameter(torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1), requires_grad = False)

        #pos = [batch size, trg len]
        # print (self.tok_embedding(trg).shape) # torch.Size([1, 361, 128])
        # print (self.scale)
        # print (self.pos_embedding(trg).shape) # torch.Size([1, 361, 128])
        # print ((self.tok_embedding(trg) * self.scale).shape)

        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))

        #trg = [batch size, trg len, hid dim]

        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        output = self.fc_out(trg)

        #output = [batch size, trg len, output dim]

        return output, attention

class DecoderLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()

        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim,
                                                                     pf_dim,
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):

        #trg = [batch size, trg len, hid dim]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, trg len]
        #src_mask = [batch size, src len]

        #self attention
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)

        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #encoder attention
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)

        #dropout, residual connection and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]

        #positionwise feedforward
        _trg = self.positionwise_feedforward(trg)

        #dropout, residual and layer norm
        trg = self.layer_norm(trg + self.dropout(_trg))

        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]

        return trg, attention

class ConvNet(nn.Module):

    def __init__(self, input_nc = 3):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_nc, 16, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(16),
            nn.ReLU( inplace=False),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU( inplace=False),
            # nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.ReLU( inplace=False),
            # nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc1 = nn.Linear(2304, 512) #dont hard code this ...
        self.fc2 = nn.Linear(512, 128)
        self.relu = nn.ReLU( inplace=False)

    def forward(self, x):


        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.fc2(out)

        return out


class Discrim(nn.Module):
    def __init__(self, input_dim = 256):
        super(Discrim, self).__init__()

        self.conv = nn.Conv1d(256, 256, 1, stride=1)
        self.conv2 = nn.Conv1d(256, 256, 1, stride = 1)
        self.last_fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU( inplace=False)
        self.dropout = nn.Dropout(0.5)

        self.norm = torch.nn.LayerNorm(256)
        self.fc = nn.Linear(128, 1)


    def forward(self, x):

        #video is of shape [batch_size x length x hid dim]
        # print (x.view(x.shape[0], -1).shape)

        # x = x.permute (0, 2, 1) #it is now [batch_size x hid_dim x length]
        # conved = self.conv(x)
        # relu_conved = self.relu(conved)

        #relu_conved = relu_conved.permute(0,2,1)
        #relu_conved = self.norm(relu_conved)
        #relu_conved = relu_conved.permute(0,2,1)


        # conved2 = self.conv2(relu_conved)
        # relu_conved2 = self.relu(conved2)


        #relu_conved2 = relu_conved2.permute(0,2,1)
        #relu_conved2 = self.norm(relu_conved2)
        #relu_conved2 = relu_conved2.permute(0,2,1)

        pooled = x.max(dim = 1)[0]
        # pooled = self.dropout(self.relu(pooled))
        out = self.last_fc(pooled)

        out = out.squeeze(1)

        return out

class Discrim_LSTM(nn.Module):
    def __init__(self):
        super(Discrim_LSTM, self).__init__()

        self.lstm = nn.LSTM(256, 128, batch_first = True)
        self.fc1 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        # self.dropout = nn.Dropout(0.25)


    def forward(self, x):

        batch_size = x.shape[0]
        x, _ = self.lstm(x)

        x = x[:,-1,:].clone()
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = x.squeeze(1)

        return x
#
#
# class Discrim_LSTM_Multi(nn.Module):
#     def __init__(self):
#         super(Discrim_LSTM_Multi, self).__init__()
#
#         self.lstm = nn.LSTM(256, 128, batch_first = True)
#         self.fc1 = nn.Linear(128, 4)
#
#         # self.dropout = nn.Dropout(0.25)
#
#
#     def forward(self, x):
#
#         batch_size = x.shape[0]
#         # x = self.dropout(x)
#         x, _ = self.lstm(x)
#
#         x = x[:,-1,:].clone()
#         x = self.fc1(x)
#
#
#         return x



class Discrim_Multi(nn.Module):
    def __init__(self):
        super(Discrim_Multi, self).__init__()
        self.conv1 = nn.Conv1d(256, 256, 1, stride=1)
        self.conv2 = nn.Conv1d(256, 256, 1, stride=1)
        self.conv3 = nn.Conv1d(256, 256, 1, stride=1)
        self.conv4 = nn.Conv1d(256, 256, 1, stride=1)
        self.last_fc = nn.Linear(256, 4)
        self.relu = nn.LeakyReLU( inplace=False)
        self.norm = torch.nn.LayerNorm(256)
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):


        # x = x.permute (0, 2, 1) #it is now [batch_size x hid_dim x length]
        #
        # conved = self.conv1(x)
        # relu_conved = self.relu(conved)
        #
        # conved2 = self.conv2(relu_conved)
        # relu_conved2 = self.relu(conved2)
        #
        # pooled = relu_conved2.max(dim = 2)[0]
        # out = self.last_fc(pooled)

        pooled = x.max(dim = 1)[0]
        # pooled = self.dropout(self.relu(pooled))
        out = self.last_fc(pooled)


        return out


class Transformer(nn.Module):
    def __init__(self,cnn, encoder, decoder, src_pad_idx, trg_pad_idx, device,
                    criterion, max_length = 70, max_pool = True):
        super().__init__()

        self.cnn = cnn
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        self.criterion = criterion
        self.SOS_TOKEN = 0
        self.EOS_TOKEN = 29
        self.PAD_TOKEN = 1
        self.max_length = max_length
        self.max_pool = max_pool


    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        src_len = src.shape[1]
        batch_size = src.shape[0]
        src_mask = torch.ones([batch_size, 1 , 1, src_len], dtype = torch.bool, device = self.device)
        # src_mask = torch.nn.Parameter( torch.ones([batch_size, 1 , 1, src_len], dtype = torch.bool ), requires_grad = False)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask

    def make_trg_mask(self, trg):

        #trg = [batch size, trg len]

        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(3)

        #trg_pad_mask = [batch size, 1, trg len, 1]

        trg_len = trg.shape[1]

        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        # trg_sub_mask = torch.nn.Parameter(torch.tril(torch.ones((trg_len, trg_len))).bool(), requires_grad = False)

        #trg_sub_mask = [trg len, trg len]

        trg_mask = trg_pad_mask & trg_sub_mask

        #trg_mask = [batch size, 1, trg len, trg len]

        return trg_mask

    def forward(self, src, trg):

        #src = [batch size, src len]
        #trg = [batch size, trg len]

        # print (src.shape)
        # exit()
        #


        #cnn_out = torch.Size([batch, source length, in-feature dimensions])
        #trg = [batch, targ length]

        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg[:,:-1].clone() )


        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)
        # enc_pooled = enc_src.max(dim=1)[0] if self.max_pool else x.mean(dim=1)

        #enc_src = [batch size, src len, hid dim]
        output, attention = self.decoder(trg[:,:-1].clone(), enc_src, trg_mask, src_mask)

        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]

        output = output.contiguous().view(-1, 30)
        trg = trg[:,1:].clone().contiguous().view(-1)

        seq_loss = self.criterion(output, trg)


        return seq_loss

def make_model(input_size, embed_size, hidden_size, output_dim, n_layers,
                dropout, device, enc_heads, dec_heads, criterion):

    # cnn = ConvNet()
    enc = Encoder(input_size, hidden_size,n_layers, enc_heads,
                    embed_size, dropout, device)
    dec = Decoder(output_dim, hidden_size, n_layers, dec_heads,
                    embed_size, dropout, device)
    cnn = ConvNet()

    model = Transformer(cnn, enc, dec, -1, 1, device, criterion) #.to(device)

    return model
#
#



class CNN_Transformer_Encoder(nn.Module):
    def __init__(self, cnn, encoder, device, src_pad_idx= -1):
        super().__init__()

        self.cnn = cnn
        self.encoder = encoder
        self.src_pad_idx = src_pad_idx
        self.device = device


    def make_src_mask(self, src):

        #src = [batch size, src len]

        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        src_len = src.shape[1]
        batch_size = src.shape[0]
        src_mask = torch.ones([batch_size, 1 , 1, src_len], dtype = torch.bool, device = self.device)
        # src_mask = torch.nn.Parameter( torch.ones([batch_size, 1 , 1, src_len], dtype = torch.bool ), requires_grad = False)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask


    def forward(self, src):

        #src = [batch size, src len]
        #trg = [batch size, trg len]

        if len(src.shape) == 5:
            '''
            This is a work around for this issue with daraparallel
            https://github.com/pytorch/pytorch/issues/33185
            https://github.com/pytorch/pytorch/issues/31460
            '''
            raw_vids = True
        else:
            raw_vids = False

        if raw_vids:
            batch_size, timesteps, C, H, W = src.size()
            c_in = src.view(batch_size * timesteps, C, H, W)
            c_out = self.cnn(c_in)
            src = c_out.view(batch_size, timesteps, -1)

        #cnn_out = torch.Size([batch, source length, in-feature dimensions])
        #trg = [batch, targ length]

        src_mask = self.make_src_mask(src)
        # trg_mask = self.make_trg_mask(trg[:,:-1])

        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]

        enc_src = self.encoder(src, src_mask)

        return enc_src, src


class Aggregate(nn.Module):
    def __init__(self, timesteps = 300, hid_dim = 256, dropout = 0.2 ):
        super(Aggregate, self).__init__()

        self.timesteps = timesteps
        self.hid_dim = hid_dim
        self.joint_norm = torch.nn.LayerNorm(256)




    def forward(self, input):

        style = input[0]
        content = input[1]
        content_trans = style + content
        content_trans = self.joint_norm(content_trans)

        return content_trans
