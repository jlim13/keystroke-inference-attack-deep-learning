import torch
import random
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, dropout):
        super().__init__()

        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim + hid_dim, hid_dim, batch_first = False)
        self.fc_out = nn.Linear(emb_dim + hid_dim * 2, output_dim)
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

        embedded = self.embedding(input)

        #embedded = [1, batch size, emb dim]

        emb_con = torch.cat((embedded, context), dim = 2)

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

        prediction = self.fc_out(output)

        #prediction = [batch size, output dim]

        return prediction, hidden

class TCNModel(torch.nn.Module):

    def __init__(self, tcn, input_size, output_size, device, criterion):
        super().__init__()

        self.tcn = tcn
        self.decoder = Decoder(output_size, input_size, input_size, 0.5)
        self.output_size = output_size
        self.device = device
        self.criterion = criterion

    def forward(self, vid, trg, teacher_forcing_ratio = 0.5):

        trg = trg.transpose(1,0)
        trg_len = trg.shape[0]
        batch_size = trg.shape[1]

        vid = vid.transpose(1,2)
        y = self.tcn(vid)
        context = y[:, :, -1].contiguous()
        context = context.unsqueeze(0)
        hidden = context
        #first input to the decoder is the <sos> tokens

        input = trg[0,:]
        # input = torch.tensor([[0]], device=self.device)

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


        # print (outputs)
        # print (losses/batch_size)
        # exit()
        outputs = outputs[1:].contiguous().view(-1, 30)
        trg = trg[1:].contiguous().view(-1)
        loss = self.criterion(outputs, trg)

        return loss

        # return losses/batch_size
