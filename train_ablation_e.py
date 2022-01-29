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
from utils import smooth_cross_entropy
import copy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(5)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def get_entropy_loss(preds):
        """
        Returns the entropy loss: negative of the entropy present in the
        input distribution
        """

        epsilon = 1e-8
        preds = torch.nn.functional.softmax(preds, dim = 1)

        return torch.mean(torch.sum(preds * torch.log(preds +epsilon), dim=1))

def cycle(iterable):
    while True:
        for x in iterable:
            yield x



def validation(dataloader, style_encoder, content_encoder, decoder, criterion, syn_or_real, film):

    style_encoder.eval()
    content_encoder.eval()
    decoder.eval()
    film.eval()

    if syn_or_real == 'real':
        is_real = True
    else:
        is_real = False

    with torch.no_grad():
        running_loss = 0.0

        for idx, x in enumerate(dataloader):

            real_vid, real_targ, real_dp = x

            real_vid = real_vid.to(device)
            real_targ = real_targ.to(device)

            encoded_style, _ = style_encoder(real_vid)
            encoded_content, _ = content_encoder(real_vid)

            # style_accuracy(style_discrim(encoded_style), syn_or_real)
            # style_accuracy(style_discrim(encoded_content), syn_or_real)
            # exit()
            encoded = film([encoded_style, encoded_content, is_real])

            # encoded = encoded_style + encoded_content
            # encoded = nn.functional.relu(encoded)
            # encoded = torch.cat ((encoded_style , encoded_content), dim = 2)
            # encoded = fused_net(encoded)
            #enc_src = [batch size, src len, hid dim]
            output, attention = decoder(real_targ[:,:-1], encoded)

            output = output.contiguous().view(-1, 30)
            real_targ = real_targ[:,1:].contiguous().view(-1)

            seq_loss = criterion(output, real_targ)

            running_loss += seq_loss.item()

    return running_loss / (len(dataloader))

def validation_content(dataloader, content_encoder, decoder, criterion, syn_or_real):

    content_encoder.eval()
    decoder.eval()


    with torch.no_grad():
        running_loss = 0.0

        for idx, x in enumerate(dataloader):

            real_vid, real_targ, real_dp = x

            real_vid = real_vid.to(device)
            real_targ = real_targ.to(device)

            encoded, _ = content_encoder(real_vid)
            # encoded = torch.cat ((encoded_content , encoded_style), dim = 2)
            #enc_src = [batch size, src len, hid dim]
            output, attention = decoder(real_targ[:,:-1], encoded)

            output = output.contiguous().view(-1, 30)
            real_targ = real_targ[:,1:].contiguous().view(-1)

            seq_loss = criterion(output, real_targ)

            running_loss += seq_loss.item()

    return running_loss / (len(dataloader))

def prediction(dataloader, content_encoder, style_encoder,
        decoder, target_alphabet , film, syn_or_real, outfname, SOS_TOKEN = 0, EOS_TOKEN = 29, MAX_LENGTH = 70, PAD_TOKEN = 1, verbose = False):

    target_alphabet = dict( (v,k) for k,v in target_alphabet.items())

    content_encoder.eval()
    style_encoder.eval()
    decoder.eval()
    film.eval()

    all_trgs = []
    all_pred_trgs = []

    all_trgs_f = []
    all_pred_trgs_f = []

    if syn_or_real == 'real':
        is_real = True
    else:
        is_real = False

    for idx , x in enumerate(dataloader):

        vid, targ, dp = x

        vid = vid.to(device)
        targ = targ.to(device)

        with torch.no_grad():
            encoded_content, _ = content_encoder(vid)
            encoded_style, _ = style_encoder(vid)

        # encoded = encoded_style + encoded_content
        encoded = film([encoded_style, encoded_content, is_real])

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

    utils.write_to_f(all_pred_trgs, all_trgs, fname = outfname )
    bleu= bleu_score(all_pred_trgs, all_trgs) #, max_n=1, weights = [1.])
    return bleu


def _init_fn(worker_id):
    np.random.seed(int(0))



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    #model hyperparameters
    parser.add_argument("--hid_dim", type = int, default = 256)
    parser.add_argument("--emb_dim", type = int, default = 128)
    parser.add_argument("--n_layers", type = int, default = 4)
    parser.add_argument("--output_dim", type = int, default = 30)
    parser.add_argument("--dropout", type = float, default = 0.1)
    parser.add_argument("--num_heads", type = int, default = 4)
    parser.add_argument("--margin", type = float, default = .2)

    #optimizer hyperparameters
    parser.add_argument("--transformer_lr", type = float, default = 0.0001)
    parser.add_argument("--content_discrim_lr", type = float, default = 0.0004)
    parser.add_argument("--joint_discrim_lr", type = float, default = 0.0004)
    parser.add_argument("--style_discrim_lr", type = float, default = 0.0004)
    parser.add_argument("--max_iters", type = int, default = 60000)
    parser.add_argument("--validation_iter", type = int, default = 100)

    #dataset hyperparameters
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument("--num_workers", type = int, default = 8)

    #data paths
    parser.add_argument("--synthetic_train_data_path", type = str, default = '/data/synthetic_preprocessed_split/train')
    parser.add_argument("--synthetic_test_data_path", type = str, default = '/data/synthetic_preprocessed_split/val')
    parser.add_argument("--real_train_data_path", type = str, default = 'data/data/train/')
    parser.add_argument("--real_test_data_path", type = str, default = 'data/data/test_split')
    parser.add_argument("--real_val_data_path", type = str, default = 'data/data/val_split/')
    parser.add_argument("--viz_data_path", type = str, default = 'viz_plots')
    parser.add_argument("--sentence_outputs_fname", type = str, default = 'sent.txt')

    #model paths
    parser.add_argument("--pretrained_synthetic_transformer_encoder", type = str, default = 'weights/synthetic_transformer/350_encoder_model.pt' )
    parser.add_argument("--pretrained_synthetic_transformer_decoder", type = str, default = 'weights/synthetic_transformer/350_decoder_model.pt' )
    parser.add_argument("--pretrained_cnn_extractor", type = str, default = 'best_target_cnn_extractor_noBN.pth')
    parser.add_argument("--output_path", type = str, default = 'weights/multi_task_disentanglement/')

    #loss terms
    parser.add_argument("--content_adversarial_loss_weight", type = float, default = 0.1)
    parser.add_argument("--style_adversarial_loss_weight", type = float, default = 0.8)
    parser.add_argument("--content_cls_loss", type = float, default = 0)
    parser.add_argument("--style_cls_loss", type = float, default = 0)
    parser.add_argument("--sequence_loss", type = float, default = 1)
    parser.add_argument("--style_discriminator_loss", type = float, default = 1)
    parser.add_argument("--content_discriminator_loss", type = float, default = 1)
    parser.add_argument("--joined_adversarial_loss", type = float, default = 0.25)
    parser.add_argument("--joined_discrim_loss", type = float, default = 1)
    args = parser.parse_args()

    print (args)


    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    if not os.path.exists(args.viz_data_path):
        os.makedirs(args.viz_data_path)

    syn_train_dataset = Raw_Video_Dataset(args.synthetic_train_data_path, synthetic = True, training = True, video_length = 300)
    syn_train_dataloader = data.DataLoader(syn_train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True, drop_last = True, worker_init_fn=_init_fn)
    syn_train_cycle = cycle(syn_train_dataloader)

    real_train_dataset = Raw_Video_Dataset(args.real_train_data_path, synthetic = False, training = True, video_length = 300)
    real_train_dataloader = data.DataLoader(real_train_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = True, drop_last = True, worker_init_fn=_init_fn)
    real_train_cycle = cycle(real_train_dataloader)

    syn_test_dataset = Raw_Video_Dataset(args.synthetic_test_data_path, synthetic = True, training = False, video_length = 300)
    syn_test_dataloader = data.DataLoader(syn_test_dataset, batch_size = 128, num_workers = args.num_workers, shuffle = False, worker_init_fn=_init_fn)

    real_val_dataset = Raw_Video_Dataset(args.real_val_data_path, synthetic = False, training = False, video_length = 300, random_seed = True)
    real_val_dataloader = data.DataLoader(real_val_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
    bleu_dataloader_val = data.DataLoader(real_val_dataset, batch_size = 1, num_workers = 0, shuffle = False)

    real_test_dataset = Raw_Video_Dataset(args.real_test_data_path, synthetic = False, training = False, video_length = 300, random_seed = True)
    real_test_dataloader = data.DataLoader(real_test_dataset, batch_size = args.batch_size, num_workers = args.num_workers, shuffle = False)
    bleu_dataloader_test = data.DataLoader(real_test_dataset, batch_size = 1, num_workers = 0, shuffle = False)

    seq_criterion = nn.CrossEntropyLoss(ignore_index = 1)
    binary_criterion = nn.BCEWithLogitsLoss()

    '''
    Style Encoder.
    This handles domain specific factors.
        1. Texture
        2. "Style" of thumb motion
    How was it typed?
    '''
    encoder = transformer.Encoder(128, args.hid_dim, args.n_layers, args.num_heads, args.emb_dim, args.dropout, device,max_length = 300)
    cnn = transformer.ConvNet()
    cnn.load_state_dict(torch.load(args.pretrained_cnn_extractor))
    style_encoder = transformer.CNN_Transformer_Encoder(cnn, encoder, device)


    '''
    Content Encoder.
    This handles the semantic meaning of the video.
    What was typed?
    '''
    encoder = transformer.Encoder(128, args.hid_dim, args.n_layers, args.num_heads, args.emb_dim, args.dropout, device, max_length = 300)
    cnn = transformer.ConvNet()
    cnn.load_state_dict(torch.load(args.pretrained_cnn_extractor))
    content_encoder = transformer.CNN_Transformer_Encoder(cnn, encoder, device)


    '''Decoder'''
    decoder = transformer.Decoder(args.output_dim, args.hid_dim, args.n_layers, args.num_heads, args.emb_dim, args.dropout, device)



    '''film to join style+ content'''
    film = transformer.FiLM()



    optimizer_cls = torch.optim.Adam(
        [
            {"params": content_encoder.parameters(), "lr": args.transformer_lr },
            {"params": style_encoder.parameters(), "lr" : args.transformer_lr },
            {"params": decoder.parameters(), "lr": args.transformer_lr },
            {"params": film.parameters(), "lr": args.transformer_lr },

        ],
        lr=args.transformer_lr, weight_decay=1e-5
        )



    '''Load weights and set up dataparallel'''


    style_encoder = nn.DataParallel(style_encoder)
    style_encoder.load_state_dict(torch.load(args.pretrained_synthetic_transformer_encoder))
    style_encoder.to(device)

    content_encoder = nn.DataParallel(content_encoder)
    content_encoder.load_state_dict(torch.load(args.pretrained_synthetic_transformer_encoder))
    content_encoder.to(device)

    decoder = nn.DataParallel(decoder)
    decoder.load_state_dict(torch.load(args.pretrained_synthetic_transformer_decoder))
    decoder.to(device)


    film = nn.DataParallel(film)
    film.to(device)

    best_bleu = -1
    best_real_val_loss = 1000


    running_seq_loss = 0
    running_rs_rc_loss = 0
    running_rs_sc_loss = 0
    running_ss_rc_loss = 0
    running_ss_sc_loss = 0


    model_iter = 'best'  # model a best
    print ("Loading weights from {}".format(model_iter))
    style_encoder.load_state_dict(torch.load(os.path.join(args.output_path, 'style_encoder_{}.pth'.format(model_iter) )))
    content_encoder.load_state_dict(torch.load(os.path.join(args.output_path, 'content_encoder_{}.pth'.format(model_iter))))
    decoder.load_state_dict(torch.load(os.path.join(args.output_path, 'decoder_{}.pth'.format(model_iter))))
    film.load_state_dict(torch.load(os.path.join(args.output_path, 'film_{}.pth'.format(model_iter) )))
    bleu = prediction(bleu_dataloader_test,content_encoder, style_encoder, decoder, real_val_dataset.alphabet, film, 'real', args.sentence_outputs_fname )
    print (bleu)
    bleu = prediction(bleu_dataloader_test,content_encoder, style_encoder, decoder, real_val_dataset.alphabet, film, 'real', args.sentence_outputs_fname )
    print (bleu)
    #
    exit()


    for train_iter in range(args.max_iters):

        optimizer_cls.zero_grad()

        synthetic_data = next(syn_train_cycle)
        real_data = next(real_train_cycle)
        syn_vid, syn_targ, _ = synthetic_data
        real_vid, real_targ, _ = real_data

        real_vid = real_vid.to(device)
        real_targ = real_targ.to(device)
        syn_vid = syn_vid.to(device)
        syn_targ = syn_targ.to(device)

        real_style, real_style_feats = style_encoder(real_vid)
        syn_style, syn_style_feats = style_encoder(syn_vid)
        real_content, real_content_feats = content_encoder(real_vid)
        syn_content, syn_content_feats = content_encoder(syn_vid)

        content_concat = torch.cat((real_content, syn_content), 0)
        style_concat = torch.cat((real_style, syn_style), 0)


        '''
        Losses on Joined space
        '''
        #

        # print (utils.print_feature_statistics(real_style), utils.print_feature_statistics(syn_style))
        # print ( utils.print_feature_statistics(real_content), utils.print_feature_statistics(syn_content))
        rs_rc_encoded = film([real_style, real_content, True ])
        rs_sc_encoded = film([real_style, syn_content,  True ])
        ss_rc_encoded = film([syn_style, real_content, False])
        ss_sc_encoded = film([syn_style, syn_content, False])

        y_1, _ = decoder(real_targ[:,:-1].clone(), rs_rc_encoded)
        y_2, _ = decoder(syn_targ[:,:-1].clone(), rs_sc_encoded)
        y_3, _ = decoder(real_targ[:,:-1].clone(), ss_rc_encoded)
        y_4, _ = decoder(syn_targ[:,:-1].clone(), ss_sc_encoded)

        y_1 = y_1.contiguous().view(-1, args.output_dim)
        y_2 = y_2.contiguous().view(-1, args.output_dim)
        y_3 = y_3.contiguous().view(-1, args.output_dim)
        y_4 = y_4.contiguous().view(-1, args.output_dim)

        real_targ_ = real_targ[:,1:].contiguous().view(-1)
        syn_targ_ = syn_targ[:,1:].contiguous().view(-1)

        rs_rc_loss = seq_criterion(y_1, real_targ_)
        rs_sc_loss = seq_criterion(y_2, syn_targ_)
        ss_rc_loss = seq_criterion(y_3, real_targ_)
        ss_sc_loss = seq_criterion(y_4, syn_targ_)

        rs_rc_labels = torch.tensor(np.full(rs_rc_encoded.size(0),0 ), requires_grad = False).to(device)
        rs_sc_labels = torch.tensor(np.full(rs_sc_encoded.size(0),1 ), requires_grad = False).to(device)
        ss_rc_labels = torch.tensor(np.full(ss_rc_encoded.size(0),2 ), requires_grad = False).to(device)
        ss_sc_labels = torch.tensor(np.full(ss_sc_encoded.size(0),3 ), requires_grad = False).to(device)



        running_rs_rc_loss += rs_rc_loss.item()
        running_rs_sc_loss += rs_sc_loss.item()
        running_ss_rc_loss += ss_rc_loss.item()
        running_ss_sc_loss += ss_sc_loss.item()

        sequence_loss = rs_rc_loss + rs_sc_loss + ss_rc_loss + ss_sc_loss


        running_seq_loss += sequence_loss.item()

        cls_loss = sequence_loss * args.sequence_loss



        cls_loss.backward()

        optimizer_cls.step()


        if (train_iter % args.validation_iter) == 1:
            print ("Running Validation")

            this_seq_loss = running_seq_loss / (args.validation_iter)
            this_rs_rc_loss = running_rs_rc_loss / (args.validation_iter)
            this_rs_sc_loss = running_rs_sc_loss / (args.validation_iter)
            this_ss_rc_loss = running_ss_rc_loss / (args.validation_iter)
            this_ss_sc_loss = running_ss_sc_loss / (args.validation_iter)

            syn_val_loss = validation(syn_test_dataloader, style_encoder, content_encoder, decoder, seq_criterion, 'synthetic', film)
            real_val_loss = validation(real_val_dataloader, style_encoder, content_encoder, decoder, seq_criterion, 'real', film)

            real_val_content_loss = validation_content(real_val_dataloader, content_encoder, decoder, seq_criterion, 'real')
            syn_val_content_loss = validation_content(syn_test_dataloader, content_encoder, decoder, seq_criterion, 'synthetic')
            bleu_val = prediction(bleu_dataloader_val, content_encoder, style_encoder, decoder, real_val_dataset.alphabet, film, 'real', outfname = args.sentence_outputs_fname )

            test_bleu = prediction(bleu_dataloader_test, content_encoder, style_encoder, decoder, real_val_dataset.alphabet, film, 'real', outfname = args.sentence_outputs_fname )


            print ("Iter : {} | Seq Loss {} |"
                     "rs_rc Loss {} | rs_sc Loss {} | ss_rc Loss {} | ss_ss Loss {} | ". \
                    format(train_iter,
                              this_seq_loss, this_rs_rc_loss,
                             this_rs_sc_loss, this_ss_rc_loss, this_ss_sc_loss
                              ))

            print ("Real val loss {} | Real val loss (content only) {} | "
                " Syn val loss {} | Syn val loss (content only) {} | "
                "Bleu Score {} | Previous Best Bleu {} | Best Real Val Loss {} |". \
                    format(real_val_loss, real_val_content_loss,
                    syn_val_loss, syn_val_content_loss, test_bleu, best_bleu, best_real_val_loss))

            print ("Saving models at {}".format(train_iter))
            content_encoder_path = os.path.join(args.output_path, 'content_encoder_{}.pth'.format(train_iter) )
            style_encoder_path = os.path.join(args.output_path, 'style_encoder_{}.pth'.format(train_iter) )
            decoder_path = os.path.join(args.output_path, 'decoder_{}.pth'.format(train_iter) )
            film_path = os.path.join(args.output_path, 'film_{}.pth'.format(train_iter))

            torch.save(content_encoder.state_dict(), content_encoder_path)
            torch.save(style_encoder.state_dict(), style_encoder_path)
            torch.save(decoder.state_dict(), decoder_path)
            torch.save(film.state_dict(), film_path)

            if test_bleu > best_bleu:
                best_bleu = test_bleu

                content_encoder_path = os.path.join(args.output_path, 'content_encoder_best.pth')
                style_encoder_path = os.path.join(args.output_path, 'style_encoder_best.pth')
                decoder_path = os.path.join(args.output_path, 'decoder_best.pth')
                film_path = os.path.join(args.output_path, 'film_best.pth')

                torch.save(content_encoder.state_dict(), content_encoder_path)
                torch.save(style_encoder.state_dict(), style_encoder_path)
                torch.save(decoder.state_dict(), decoder_path)
                torch.save(film.state_dict(), film_path)


            if real_val_loss < best_real_val_loss:
                best_real_val_loss = real_val_loss



            style_plot_fname = os.path.join(args.viz_data_path, 'style_tsne_{}.png'.format(train_iter))
            content_plot_fname = os.path.join(args.viz_data_path, 'content_tsne_{}.png'.format(train_iter))

            utils.plot_tsne(real_val_dataloader, syn_test_dataloader, style_encoder, filename = style_plot_fname )
            utils.plot_tsne(real_val_dataloader, syn_test_dataloader, content_encoder, filename = content_plot_fname)

            style_encoder.train()
            content_encoder.train()
            decoder.train()
            film.train()

            running_seq_loss = 0
            running_rs_rc_loss = 0
            running_rs_sc_loss = 0
            running_ss_rc_loss = 0
            running_ss_sc_loss = 0
