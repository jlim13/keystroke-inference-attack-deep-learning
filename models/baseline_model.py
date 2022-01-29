import torch
import random

class BaselineModelConv(torch.nn.Module):

    def __init__(self, encoder, decoder, criterion,  EOS_TOKEN = 29, SOS_TOKEN = 0, PAD_TOKEN = 1):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.SOS_TOKEN = SOS_TOKEN
        self.EOS_TOKEN = EOS_TOKEN
        self.PAD_TOKEN = PAD_TOKEN
        self.max_length = self.decoder.max_length
        self.output_dim = self.decoder.output_dim
        self.criterion = criterion
        # self.discriminator = discriminator

    def forward(self, vid, trg):
        encoder_conved, encoder_combined = self.encoder(vid)
        #encoder_conved = [batch size, src len, emb dim]
        #encoder_combined = [batch size, src len, emb dim]

        #calculate predictions of next words
        #output is a batch of predictions for each word in the trg sentence
        #attention a batch of attention scores across the src sentence for
        #  each word in the trg sentence
        output, attention = self.decoder(trg[:,:-1], encoder_conved, encoder_combined)

        output = output.contiguous().view(-1, 30)
        trg = trg[:,1:].contiguous().view(-1)

        loss = self.criterion(output, trg)

        return loss
