#!/usr/bin/env python
# coding: utf-8

# In[1]:


import time
import argparse
import math
import os
import nltk
import torch
import torch.nn as nn
from torch import optim
from torch.autograd import Variable
import numpy
import matplotlib
from matplotlib import pyplot as plt


# In[2]:


from model import make_model, Classifier, NoamOpt, LabelSmoothing, fgim_attack
from data import (prepare_data,
                  non_pair_data_loader, 
                  get_cuda,
                  pad_batch_seuqences,
                  subsequent_mask,
                  id2text_sentence,
                  to_var,
                  calc_bleu,
                  load_human_answer,
                  load_word_dict_info)


# In[3]:


# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# In[4]:


def get_args():
    ######################################################################################
    #  Environmental parameters
    ######################################################################################
    parser = argparse.ArgumentParser(description="Here is your model discription.")
    parser.add_argument('--id_pad', type=int, default=0, help='')
    parser.add_argument('--id_unk', type=int, default=1, help='')
    parser.add_argument('--id_bos', type=int, default=2, help='')
    parser.add_argument('--id_eos', type=int, default=3, help='')

    ######################################################################################
    #  File parameters
    ######################################################################################
    parser.add_argument('--task', type=str, default='yelp', help='Specify datasets.')
    parser.add_argument('--word_to_id_file', type=str, default='', help='')
    parser.add_argument('--data_path', type=str, default='', help='')

    ######################################################################################
    #  Model parameters
    ######################################################################################
    parser.add_argument('--word_dict_max_num', type=int, default=5, help='')
    parser.add_argument('--batch_size', type=int, default=128, help='')
    parser.add_argument('--max_sequence_length', type=int, default=128)
    parser.add_argument('--num_layers_AE', type=int, default=2)
    parser.add_argument('--transformer_model_size', type=int, default=256)
    parser.add_argument('--transformer_ff_size', type=int, default=1024)
    parser.add_argument('--latent_size', type=int, default=256)
    parser.add_argument('--word_dropout', type=float, default=1.0)
    parser.add_argument('--embedding_dropout', type=float, default=0.5)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--label_size', type=int, default=1)

    args = parser.parse_args(['--task=custom'])

    args.if_load_from_checkpoint = True
    args.checkpoint_name = "1557667911"
    args.current_save_path = 'save/%s/' % args.checkpoint_name
    args.id_to_word, args.vocab_size = load_word_dict_info('word_to_id.txt', args.word_dict_max_num)
    return args


# In[5]:


def tokenize(sentence):
    with open('word_to_id.txt', 'r') as fin:
        word_dict = {line.strip().split('\t')[0]: idx for idx, line in enumerate(fin.readlines())}
    word_list = nltk.word_tokenize(sentence.lower())
    return list(map(word_dict.__getitem__, word_list))


# In[6]:


def get_models(args):
    ae_model = get_cuda(make_model(d_vocab=args.vocab_size,
                                   N=args.num_layers_AE,
                                   d_model=args.transformer_model_size,
                                   latent_size=args.latent_size,
                                   d_ff=args.transformer_ff_size))
    dis_model = get_cuda(Classifier(latent_size=args.latent_size, output_size=args.label_size))
    ae_model.load_state_dict(torch.load(args.current_save_path + 'ae_model_params.pkl', map_location=torch.device('cpu')))
    dis_model.load_state_dict(torch.load(args.current_save_path + 'dis_model_params.pkl', map_location=torch.device('cpu')))
    return ae_model, dis_model


# In[7]:


class DataLoader():
    def __init__(self, batch_size, id_bos, id_eos, id_unk, max_sequence_length, vocab_size):
        self.sentences_batches = []
        self.labels_batches = []

        self.src_batches = []
        self.src_mask_batches = []
        self.tgt_batches = []
        self.tgt_y_batches = []
        self.tgt_mask_batches = []
        self.ntokens_batches = []

        self.num_batch = 0
        self.batch_size = batch_size
        self.pointer = 0
        self.id_bos = id_bos
        self.id_eos = id_eos
        self.id_unk = id_unk
        self.max_sequence_length = max_sequence_length
        self.vocab_size = vocab_size

    def create_batches(self, input_tokens, input_label):
        self.data_label_pairs = [[input_tokens, [input_label]]]

        # Split batches
        if self.batch_size == None:
            self.batch_size = len(self.data_label_pairs)
        self.num_batch = int(len(self.data_label_pairs) / self.batch_size)
        for _index in range(self.num_batch):
            item_data_label_pairs = self.data_label_pairs[_index*self.batch_size:(_index+1)*self.batch_size]
            item_sentences = [_i[0] for _i in item_data_label_pairs]
            item_labels = [_i[1] for _i in item_data_label_pairs]

            batch_encoder_input, batch_decoder_input, batch_decoder_target,             batch_encoder_length, batch_decoder_length = pad_batch_seuqences(
                item_sentences, self.id_bos, self.id_eos, self.id_unk, self.max_sequence_length, self.vocab_size,)

            src = get_cuda(torch.tensor(batch_encoder_input, dtype=torch.long))
            tgt = get_cuda(torch.tensor(batch_decoder_input, dtype=torch.long))
            tgt_y = get_cuda(torch.tensor(batch_decoder_target, dtype=torch.long))

            src_mask = (src != 0).unsqueeze(-2)
            tgt_mask = self.make_std_mask(tgt, 0)
            ntokens = (tgt_y != 0).data.sum().float()

            self.sentences_batches.append(item_sentences)
            self.labels_batches.append(get_cuda(torch.tensor(item_labels, dtype=torch.float)))
            self.src_batches.append(src)
            self.tgt_batches.append(tgt)
            self.tgt_y_batches.append(tgt_y)
            self.src_mask_batches.append(src_mask)
            self.tgt_mask_batches.append(tgt_mask)
            self.ntokens_batches.append(ntokens)

        self.pointer = 0

    def make_std_mask(self, tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask

    def next_batch(self):
        """take next batch by self.pointer"""
        this_batch_sentences = self.sentences_batches[self.pointer]
        this_batch_labels = self.labels_batches[self.pointer]

        this_src = self.src_batches[self.pointer]
        this_src_mask = self.src_mask_batches[self.pointer]
        this_tgt = self.tgt_batches[self.pointer]
        this_tgt_y = self.tgt_y_batches[self.pointer]
        this_tgt_mask = self.tgt_mask_batches[self.pointer]
        this_ntokens = self.ntokens_batches[self.pointer]

        self.pointer = (self.pointer + 1) % self.num_batch
        return this_batch_sentences, this_batch_labels,                this_src, this_src_mask, this_tgt, this_tgt_y,                this_tgt_mask, this_ntokens

    def reset_pointer(self):
        self.pointer = 0


# In[8]:


def create_batch(args, sentence, epsilon):
    eval_data_loader = DataLoader(batch_size=1,
                                  id_bos=args.id_bos,
                                  id_eos=args.id_eos,
                                  id_unk=args.id_unk,
                                  max_sequence_length=args.max_sequence_length,
                                  vocab_size=args.vocab_size)
    eval_data_loader.create_batches(tokenize(sentence), 0 if epsilon > 0 else 1)
    return eval_data_loader.next_batch()


# In[22]:


def predict(args, ae_model, dis_model, batch, epsilon):
    (batch_sentences, 
     tensor_labels, 
     tensor_src,
     tensor_src_mask,
     tensor_tgt,
     tensor_tgt_y,
     tensor_tgt_mask,
     tensor_ntokens) = batch
    
    ae_model.eval()
    dis_model.eval()
    latent, out = ae_model.forward(tensor_src, tensor_tgt, tensor_src_mask, tensor_tgt_mask)
    generator_text = ae_model.greedy_decode(latent,
                                            max_len=args.max_sequence_length,
                                            start_id=args.id_bos)
    print(id2text_sentence(tensor_tgt_y[0], args.id_to_word))
    print(id2text_sentence(generator_text[0], args.id_to_word))
    target = get_cuda(torch.tensor([[1.0]], dtype=torch.float))
    if tensor_labels[0].item() > 0.5:
        target = get_cuda(torch.tensor([[0.0]], dtype=torch.float))

    dis_criterion = nn.BCELoss(size_average=True)

    data = to_var(latent.clone())  # (batch_size, seq_length, latent_size)
    data.requires_grad = True
    output = dis_model.forward(data)
    loss = dis_criterion(output, target)
    dis_model.zero_grad()
    loss.backward()
    data_grad = data.grad.data
    data = data - epsilon * data_grad

    generator_id = ae_model.greedy_decode(data,
                                          max_len=args.max_sequence_length,
                                          start_id=args.id_bos)
    return id2text_sentence(generator_id[0], args.id_to_word)


# In[23]:


def sentiment_transfer(sentence, epsilon):
    args = get_args()
    ae_model, dis_model = get_models(args)
    batch = create_batch(args, sentence, epsilon)
    return predict(args, ae_model, dis_model, batch, abs(epsilon))


# In[42]:


if __name__ == '__main__':
    sentence = "This restaurant is the most horrible place I've ever been to. Everything is disgusting and dirty."
    print(sentiment_transfer(sentence, 5))


# In[ ]:




