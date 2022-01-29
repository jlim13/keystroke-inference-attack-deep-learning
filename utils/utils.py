from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from queue import PriorityQueue
import operator
from matplotlib.lines import Line2D


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.savefig("Gradient flow")
    plt.clf()

import math

class LabelSmoothingLoss(nn.Module):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """
    def __init__(self, label_smoothing, tgt_vocab_size= 30, ignore_index=1):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__()

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))

        self.confidence =  1.0 - label_smoothing
        # self.confidence = torch.nn.Parameter(torch.FloatTensor([1.0 - label_smoothing]), requires_grad = False)

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size x n_classes
        target (LongTensor): batch_size
        """

        model_prob = self.one_hot.repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return torch.nn.functional.kl_div(output, model_prob, reduction='sum')

def write_to_f( list_of_pred, list_of_gt, fname):

    #gt, pred
    with open(fname, 'w+') as f:
        for gt, pred in zip(list_of_gt, list_of_pred):
            f.write(' '.join(gt[0]))
            f.write(', ')
            f.write(' '.join(pred))
            f.write('\n')


def plot_tsne(real_data, synthetic_data, encoder, filename):

    with torch.no_grad():

        encoder.eval()

        encoded_tensors = []
        labels = [] #0 is real, 1 is synthetic
        color_dict = {0: 'r', 1: 'b'}

        for x in real_data:

            real_vid, _, _ = x
            batch_size = real_vid.shape[0]

            real_encoded, _= encoder(real_vid) #, raw_vids = True)
            # real_encoded = real_encoded.mean(dim=1).detach().cpu().numpy()
            # real_encoded = real_encoded.max(dim=1)[0].detach().cpu().numpy()
            real_encoded = real_encoded[:,0].detach().cpu().numpy()
            encoded_tensors.append(real_encoded)
            labels.append([0] * batch_size)

        syn_ct = 0
        for idx, x in enumerate(synthetic_data):

            if idx < 5: continue

            syn_vid, _, _ = x
            batch_size = syn_vid.shape[0]
            syn_encoded, _ = encoder(syn_vid) #, raw_vids = False)
            # syn_encoded = syn_encoded.mean(dim=1).detach().cpu().numpy()
            # syn_encoded = syn_encoded.max(dim=1)[0].detach().cpu().numpy()
            syn_encoded = syn_encoded[:,0].detach().cpu().numpy()
            encoded_tensors.append(syn_encoded)
            labels.append([1] * batch_size)

            syn_ct += batch_size

            if syn_ct > 100:
                break

        encoded_tensors = np.vstack(encoded_tensors)
        labels = np.asarray([item for sublist in labels for item in sublist])

        X_embedded = TSNE().fit_transform(encoded_tensors)

        for label in np.unique(labels):
            label_idxs = (label == labels)
            color = color_dict[label]

            these_pts = X_embedded[label_idxs]

            xs = these_pts[:,0]
            ys = these_pts[:,1]
            colors = [color] * len(ys)
            if label == 0:
                label_plt = 'real-life'
            else:
                label_plt = 'synthetic'
            plt.scatter(xs, ys, c = colors, label=label_plt)
            plt.legend(loc = 'lower left')
            plt.axis('off')


            # if label == 0:
            #     fname = '{}_real_pts_latex.txt'.format(filename)
            # else:
            #     fname = '{}_syn_pts_latex.txt'.format(filename)
            #
            # with open(fname, 'w+') as f:
            #
            #     f.write('x y\n')
            #     for x, y in zip(xs, ys):
            #         f.write(str(x) + ' ' +str(y) + '\n')


        plt.savefig(filename)
        plt.clf()




def print_feature_statistics(features):

    '''
    Min
    Max
    Mean
    Std
    '''

    tensor_mean = torch.mean(features)
    tensor_std = torch.std(features)
    tensor_min = torch.min(features)
    tensor_max = torch.max(features)
    statistics = {
        'Mean': tensor_mean.item(),
        'Std': tensor_std.item(),
        'Min': tensor_min.item(),
        'Max': tensor_max.item()
    }
    return statistics

def naive_beam_decode(content_encoder, style_encoder, decoder, iterator, max_length,
    target_alphabet, device, PAD_TOKEN = 1, EOS_TOKEN = 29, SOS_TOKEN = 0, top_k = 1000):

    target_alphabet = dict( (v,k) for k,v in target_alphabet.items())

    for idx ,batch in enumerate(iterator):

        '''
        Create a prediction matrix that is phrase_length x num_tokens
        '''

        vid, targ, dp = batch

        vid = vid.to(device)
        targ = targ.to(device)

        with torch.no_grad():
            encoded_content = content_encoder(vid)
            encoded_style = style_encoder(vid)

        encoded = encoded_style + encoded_content
        trg_indexes = [SOS_TOKEN] # sos token

        prediction_matrix = []

        for i in range(max_length):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

            with torch.no_grad():
                output, attention = decoder(trg_tensor, encoded)

            pred_token = output.argmax(2)[:,-1].item()

            prediction_values = output[:,-1].detach().cpu()
            prediction_values = torch.nn.functional.softmax(prediction_values, dim =1 ).numpy()
            prediction_matrix.append(prediction_values)

            trg_indexes.append(pred_token)

            # if pred_token == EOS_TOKEN:
            #     break

        predictions = np.stack ((prediction_matrix)).squeeze(1)

        output_sequences = [([], 0)]

        #looping through all the predictions
        for token_probs in predictions:
            new_sequences = []

            #append new tokens to old sequences and re-score
            for old_seq, old_score in output_sequences:
                for char_index in range(len(token_probs)):
                    new_seq = old_seq + [char_index]
                    #considering log-likelihood for scoring
                    new_score = old_score + math.log(token_probs[char_index])
                    new_sequences.append((new_seq, new_score))

            #sort all new sequences in the de-creasing order of their score
            output_sequences = sorted(new_sequences, key = lambda val: val[1], reverse = True)

            #select top-k based on score
            # *Note- best sequence is with the highest score
            # translations = []
            #
            # for x in output_sequences:
            #     translation = []
            #     for ch in output_sequences:
            #         if i == PAD_TOKEN or i == EOS_TOKEN:
            #             continue
            #         translation.append(target_alphabet[i])
            #     translations.append(translation)

            output_sequences = output_sequences[:top_k]
        translations =  []

        for _ in output_sequences:
            translation = []
            sent = _[0]
            for ch in sent:

                if ch == PAD_TOKEN or ch == EOS_TOKEN:
                    continue
                translation.append(target_alphabet[ch])
            translation = ''.join(translation)
            translations.append(translation)
        print (translations)
        exit()


class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        '''

        :param wordId:
        :param logProb:
        :param length:
        '''
        self.prevNode = previousNode

        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward

def beam_decode(content_encoder, style_encoder, decoder, iterator, max_length,
            target_alphabet, device, PAD_TOKEN = 1, EOS_TOKEN = 29, SOS_TOKEN =0 ):
    '''
    :param target_tensor: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
    :param decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
    :param encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
    :return: decoded_batch
    '''

    target_alphabet = dict( (v,k) for k,v in target_alphabet.items())

    beam_width = 2
    topk = 10 # how many sentence do you want to generate
    decoded_batch = []

    for idx ,batch in enumerate(iterator):

        vid, targ, dp = batch

        vid = vid.to(device)
        targ = targ.to(device)

        with torch.no_grad():
            encoded_content = content_encoder(vid)
            encoded_style = style_encoder(vid)

        encoded = encoded_style + encoded_content


        # Start with the start of the sentence token
        trg_idxs = []

        # Number of sentence to generate
        endnodes = []
        number_required = min((topk + 1), topk - len(endnodes))


        # starting node -  hidden vector, previous node, word id, logp, length
        node = BeamSearchNode(previousNode = None, wordId = SOS_TOKEN, logProb = 0, length = 1)
        nodes = PriorityQueue()

        # start the queue
        nodes.put((-node.eval(), node))
        qsize = 1

        # start beam search
        step = 0
        while True:
            # give up when decoding takes too long
            if qsize > 10000: break

            # fetch the best node
            score, n = nodes.get()
            decoder_input = n.wordid
            trg_idxs.append(decoder_input)

            trg_tensor = torch.LongTensor(trg_idxs).unsqueeze(0).to(device)

            if n.wordid == EOS_TOKEN and n.prevNode != None:
                endnodes.append((score, n))
                # if we reached maximum # of sentences required
                if len(endnodes) >= number_required:
                    break
                else:
                    continue

            # decode for one step using decoder
            output, attention = decoder(trg_tensor, encoded)

            # PUT HERE REAL BEAM SEARCH OF TOP

            log_prob, indexes = torch.topk(output, k = beam_width, dim = 2)

            log_prob = log_prob[:,-1]

            indexes = indexes[:,-1]
            nextnodes = []


            for new_k in range(beam_width):

                decoded_t = indexes[0][new_k].view(1, -1)
                log_p = log_prob[0][new_k].item()
                node = BeamSearchNode( previousNode = n, wordId = decoded_t.item(),
                                    logProb = n.logp + log_p, length = n.leng + 1)

                score = -node.eval()

                nextnodes.append((score, node))

            # put them into queue
            for i in range(len(nextnodes)):
                score, nn = nextnodes[i]
                nodes.put((score, nn))
                # increase qsize

            qsize += len(nextnodes) - 1

            step += 1

        # choose nbest paths, back trace them
        if len(endnodes) == 0:
            endnodes = [nodes.get() for _ in range(topk)]

        utterances = []

        for score, n in sorted(endnodes, key=operator.itemgetter(0)):
            utterance = []
            utterance.append(n.wordid)

            # back trace
            while n.prevNode != None:
                n = n.prevNode
                utterance.append(n.wordid)

            utterance = utterance[::-1]
            utterances.append(utterance)

        decoded_batch.append(utterances)

        for u in utterances:
            translation = [target_alphabet[x] for x in u][1:-1]
            translation = ''.join(translation)
            print (translation)

    return decoded_batch
