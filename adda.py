import os
import argparse
import time
import random
import math
import itertools

import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from torch import optim
from tqdm import tqdm
from torchtext.data.metrics import bleu_score

from dataset import Raw_Video_Dataset
from models import transformer
from utils import utils


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
np.random.seed(0)

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def validation(dataloader, syn_encoder, decoder, criterion):

    syn_encoder.eval()
    decoder.eval()

    with torch.no_grad():
        running_loss = 0.0

        for idx, x in enumerate(dataloader):

            real_vid, real_targ, real_dp = x

            real_vid = real_vid.to(device)
            real_targ = real_targ.to(device)

            encoded, _ = syn_encoder(real_vid)

            output, _ = decoder(real_targ[:,:-1], encoded) #, trg_mask, src_mask)

            output = output.contiguous().view(-1, 30)
            real_targ = real_targ[:,1:].contiguous().view(-1)

            seq_loss = criterion(output, real_targ)

            running_loss += seq_loss.item()

    return running_loss / (len(dataloader))

def prediction(dataloader, encoder, out_fname,
        decoder, target_alphabet ,SOS_TOKEN = 0, EOS_TOKEN = 29,
        MAX_LENGTH = 70, PAD_TOKEN = 1, verbose = False):

    target_alphabet = dict( (v,k) for k,v in target_alphabet.items())

    encoder.eval()
    decoder.eval()

    all_trgs = []
    all_pred_trgs = []

    all_trgs_f = []
    all_pred_trgs_f = []

    for idx , x in enumerate(dataloader):

        vid, targ, dp = x

        vid = vid.to(device)
        targ = targ.to(device)

        with torch.no_grad():
            encoded, _ = encoder(vid)

        trg_indexes = [SOS_TOKEN] # sos token

        for i in range(MAX_LENGTH):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

            with torch.no_grad():
                output, attention = decoder(trg_tensor, encoded)


            pred_token = output.argmax(2)[:,-1].item()

            trg_indexes.append(pred_token)

            if pred_token == EOS_TOKEN:
                break

        trg_tokens = [target_alphabet[i] for i in trg_indexes]
        trg_tokens.pop(0)
        trg_tokens.pop(-1)
        trg_tokens_joined = ''.join(trg_tokens) #gt

        trg_np = targ.squeeze().detach().cpu().numpy().tolist()
        trg_english = [target_alphabet[i] for i in trg_np if not i == PAD_TOKEN] #prediction
        trg_english.pop(0)
        trg_english.pop(-1)
        trg_english_joined = ''.join(trg_english)

        all_trgs.append([trg_tokens_joined.split(' ')])
        all_pred_trgs.append(trg_english_joined.split(' '))

        all_trgs_f.append(trg_tokens_joined)
        all_pred_trgs_f.append(trg_english_joined)

        if verbose:
            print ("Predicted sentence: {0}".format(trg_tokens_joined))
            print ("Ground truth sentence: {0}".format(trg_english_joined))

    utils.write_to_f(all_pred_trgs, all_trgs, fname = out_fname)
    bleu= bleu_score(all_pred_trgs, all_trgs, max_n=1, weights = [1.] )
    return bleu



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #model hyperparameters
    parser.add_argument("--hid_dim", type = int, default = 256)
    parser.add_argument("--emb_dim", type = int, default = 128)
    parser.add_argument("--n_layers", type = int, default = 4)
    parser.add_argument("--output_dim", type = int, default = 30)
    parser.add_argument("--dropout", type = float, default = 0.1)
    parser.add_argument("--num_heads", type = int, default = 4)

    #optimizer hyperparameters
    parser.add_argument("--transformer_lr", type = float, default = 0.0001)
    parser.add_argument("--style_discrim_lr", type = float, default = 0.0005)
    parser.add_argument("--max_iters", type = int, default = 30000)
    parser.add_argument("--validation_iter", type = int, default = 50)

    #dataset hyperparameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--num_workers", type = int, default = 8)

    #data paths
    parser.add_argument("--synthetic_train_data_path", type = str, default = '/data/synthetic_preprocessed_split/train')
    parser.add_argument("--synthetic_test_data_path", type = str, default = '/data/synthetic_preprocessed_split/val')
    parser.add_argument("--real_train_data_path", type = str, default = 'data/data/train')
    parser.add_argument("--real_test_data_path", type = str, default = 'data/data/test/')
    parser.add_argument('--output_path', type = str, default = 'weights/adda')

    #model paths
    parser.add_argument("--pretrained_synthetic_transformer_encoder", type = str, default = 'weights/synthetic_transformer/350_encoder_model.pt' )
    parser.add_argument("--pretrained_synthetic_transformer_decoder", type = str, default = 'weights/synthetic_transformer/350_decoder_model.pt' )
    parser.add_argument("--pretrained_cnn_extractor", type = str, default = 'domain_invariant_net/weights/adda/semisupervised_discrimlr0.0003_cnnlr0.0003_clslr0.0002_K15/best_target_cnn_extractor_.pth')

    args = parser.parse_args()
    print (args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    out_fname = 'adda_outputs.txt'

    syn_train_dataset = Raw_Video_Dataset(args.synthetic_train_data_path, synthetic = True, training = True, video_length = 300)
    syn_train_dataloader = data.DataLoader(syn_train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True, drop_last = True)
    syn_train_cycle = cycle(syn_train_dataloader)

    real_train_dataset = Raw_Video_Dataset(args.real_train_data_path, synthetic = False, training = True, video_length = 300)
    real_train_dataloader = data.DataLoader(real_train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True, drop_last = True)
    real_train_cycle = cycle(real_train_dataloader)

    syn_test_dataset = Raw_Video_Dataset(args.synthetic_test_data_path, synthetic = True, training = False, video_length = 300)
    syn_test_dataloader = data.DataLoader(syn_test_dataset, batch_size = 128, num_workers = args.num_workers, shuffle = False)

    real_test_dataset = Raw_Video_Dataset(args.real_test_data_path, synthetic = False, training = False, video_length = 300, random_seed = True)
    real_test_dataloader = data.DataLoader(real_test_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
    bleu_dataloader = data.DataLoader(real_test_dataset, batch_size = 1, num_workers = 0, shuffle = False)

    seq_criterion = nn.CrossEntropyLoss(ignore_index = 1)
    binary_criterion = nn.BCEWithLogitsLoss()

    '''
    Encoder
    '''
    encoder = transformer.Encoder(128, args.hid_dim, args.n_layers, args.num_heads, args.emb_dim, args.dropout, device)
    cnn = transformer.ConvNet()
    syn_encoder = transformer.CNN_Transformer_Encoder(cnn, encoder, device)
    syn_encoder = nn.DataParallel(syn_encoder)
    syn_encoder.load_state_dict(torch.load(args.pretrained_synthetic_transformer_encoder))
    syn_encoder.module.cnn.load_state_dict(torch.load(args.pretrained_cnn_extractor))
    syn_encoder.to(device)


    decoder = transformer.Decoder(args.output_dim, args.hid_dim, args.n_layers, args.num_heads, args.emb_dim, args.dropout, device)
    decoder = nn.DataParallel(decoder)
    decoder.load_state_dict(torch.load(args.pretrained_synthetic_transformer_decoder))
    decoder.to(device)

    style_discrim = transformer.Discrim()
    style_discrim = nn.DataParallel(style_discrim)
    style_discrim.to(device)

    '''Optimizers below'''
    optimizer_cls = torch.optim.Adam( itertools.chain( syn_encoder.parameters(),
                                        decoder.parameters(),
                                        ),
                                        lr = args.transformer_lr)
    optimizer_style_discrim = torch.optim.Adam( style_discrim.parameters(), lr = args.style_discrim_lr )

    running_seq_loss = 0
    running_discrim_loss = 0
    running_adv_loss = 0

    style_tgt_labels = torch.tensor(np.ones((args.batch_size )), requires_grad = False).float().to(device)
    style_src_labels = torch.tensor(np.zeros((args.batch_size )), requires_grad = False).float().to(device)

    style_labels = torch.cat((style_tgt_labels, style_src_labels ), 0)
    adv_labels = torch.cat((style_src_labels, style_tgt_labels), 0)

    best_bleu = -1

    for train_iter in range(args.max_iters):

        synthetic_data = next(syn_train_cycle)
        real_data = next(real_train_cycle)
        syn_vid, syn_targ, _ = synthetic_data
        real_vid, real_targ, _ = real_data

        real_vid = real_vid.to(device)
        real_targ = real_targ.to(device)
        syn_vid = syn_vid.to(device)
        syn_targ = syn_targ.to(device)

        optimizer_cls.zero_grad()
        optimizer_style_discrim.zero_grad()

        real_encoded, _ = syn_encoder(real_vid)
        syn_encoded, _ = syn_encoder(syn_vid)

        encoded_concat = torch.cat((real_encoded, syn_encoded), 0)

        discriminator_predictions = style_discrim(encoded_concat.detach()) #detach when training discriminator
        discriminator_loss = binary_criterion(discriminator_predictions, style_labels)

        adversarial_predictions = style_discrim(encoded_concat)
        adversarial_loss = binary_criterion(adversarial_predictions, adv_labels)

        real_preds, _ = decoder(real_targ[:,:-1], real_encoded)
        real_preds = real_preds.contiguous().view(-1, args.output_dim)
        real_targ = real_targ[:,1:].contiguous().view(-1)

        syn_preds, _ = decoder(syn_targ[:,:-1], syn_encoded)
        syn_preds = syn_preds.contiguous().view(-1, args.output_dim)
        syn_targ = syn_targ[:,1:].contiguous().view(-1)

        real_iter_loss = seq_criterion(real_preds, real_targ)
        syn_iter_loss = seq_criterion(syn_preds, syn_targ)
        cls_loss = real_iter_loss + adversarial_loss + syn_iter_loss

        discriminator_loss.backward(retain_graph = True)
        cls_loss.backward()

        optimizer_style_discrim.step()
        optimizer_cls.step()

        running_seq_loss += real_iter_loss.item()
        running_discrim_loss += discriminator_loss.item()
        running_adv_loss += adversarial_loss.item()


        if (train_iter % args.validation_iter) == 1:
            print ("Running Validation")


            this_seq_loss = running_seq_loss / args.validation_iter
            this_discrim_loss = running_discrim_loss / args.validation_iter
            this_adv_loss = running_adv_loss / args.validation_iter

            real_val_loss = validation(real_test_dataloader,syn_encoder, decoder, seq_criterion)
            bleu = prediction(bleu_dataloader, syn_encoder, out_fname, decoder, real_test_dataset.alphabet )

            if bleu > best_bleu:
                best_bleu = bleu

                encoder_path = os.path.join(args.output_path, 'encoder_best.pth')
                decoder_path = os.path.join(args.output_path, 'decoder_best.pth')

                torch.save(syn_encoder.state_dict(), encoder_path)
                torch.save(decoder.state_dict(), decoder_path)

            print ("Real train loss {} | Real val loss {} | Bleu-1 Score {} | "
                " Best Bleu-1 {} | Adv Loss {} | Discrim Loss {} |".\
                format(this_seq_loss, real_val_loss,
                bleu, best_bleu, this_adv_loss, this_discrim_loss))


            syn_encoder.train()
            decoder.train()

            running_seq_loss = 0
            running_adv_loss = 0
            running_discrim_loss = 0
