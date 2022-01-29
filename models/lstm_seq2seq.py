import torch.nn as nn
import torch
import random
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, dropout, num_layers ):
        super().__init__()

        self.hid_dim = hid_dim
        self.num_layers = num_layers
        self.rnn = nn.GRU(256, hid_dim, num_layers = self.num_layers, batch_first = False)
        self.linear = nn.Linear(2, 256)
        self.dropout = nn.Dropout(0.5)

    def forward(self, src):

        #src = [src len, batch size]

        #embedded = [src len, batch size, emb dim]
        src = F.relu(self.dropout(self.linear(src)))

        outputs, hidden = self.rnn(src) #no cell state!

        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        # outputs = outputs[:, :, :self.hid_dim] + outputs[:, : ,self.hid_dim:]

        #outputs are always from the top hidden layer

        batch_size = outputs.size(1)

        hidden = hidden.view(self.num_layers,1, batch_size, self.hid_dim)
        hidden = hidden[-1]


        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout, num_layers):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim , hid_dim, num_layers = 1)


        # self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
        self.fc_1 = nn.Linear(emb_dim + hid_dim * 2, 256)
        self.fc_2 = nn.Linear(256, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, context):

        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #context = [n layers * n directions, batch size, hid dim]

        #n layers and n directions in the decoder will both always be 1, therefore:
        #hidden = [1, batch size, hid dim]
        #context = [1, batch size, hid dim]

        input = input.unsqueeze(0)

        #input = [1, batch size]
        embedded = F.relu(self.embedding(input))

        #embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim = 2) #dim = 2

        #emb_con = [1, batch size, emb dim + hid dim]

        output, hidden = self.rnn(emb_con, hidden)

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]

        #seq len, n layers and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [1, batch size, hid dim]

        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim = 1)

        #output = [batch size, emb dim + hid dim * 2]
        # print (output.shape)
        # prediction = self.fc_out(output)
        output = F.relu(self.fc_1(output))
        prediction = self.fc_2(output)
        # print (prediction.shape)
        # exit()
        #prediction = [batch size, output dim]

        return prediction, hidden


class Model(torch.nn.Module):

    def __init__(self, input_dim, emb_dim, hid_dim, output_size,
                    device, criterion, dropout, num_layers ):
        super().__init__()

        self.encoder = Encoder(input_dim = 2, hid_dim = hid_dim, dropout= dropout,num_layers = num_layers )
        self.decoder = Decoder(output_size, emb_dim, hid_dim, dropout = dropout, num_layers = 1)
        self.output_size = output_size
        self.device = device
        self.criterion = criterion

    def forward(self, vid, trg, teacher_forcing_ratio = 0.55):


        trg = trg.transpose(1,0)
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]

        vid = vid.transpose(1,0)

        context = self.encoder(vid)
        hidden = context.clone()
        #first input to the decoder is the <sos> tokens

        input = trg[0,:]

        outputs = torch.zeros(trg_len, batch_size, self.output_size).to(self.device)

        for t in range(1, trg_len):

            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden, context)
            # losses += self.criterion(output, trg[t] )
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1



        outputs = outputs[1:].contiguous().view(-1, 30)
        trg = trg[1:].contiguous().view(-1)

        loss = self.criterion(outputs, trg)

        return loss
