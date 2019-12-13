# coding: utf8
"""Conll training algorithm"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function

import os
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
from torch.nn.utils.rnn import pack_padded_sequence

from transformers import DistilBertModel, BertForSequenceClassification


class Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, H1, H2, H3, D_pair_in, D_single_in, dropout=0.5,
                 bert_model='distilbert-base-uncased', D_bert=768, finetune_bert=False,
                 bert_pretrained_dir_path=None):
        super(Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.pair_top = nn.Sequential(nn.Linear(D_pair_in + D_bert, H1), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(H1, H2), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(H2, H3), nn.ReLU(), nn.Dropout(dropout),
                                      nn.Linear(H3, 1),
                                      nn.Linear(1, 1))
        self.single_top = nn.Sequential(nn.Linear(D_single_in, H1), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H1, H2), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H2, H3), nn.ReLU(), nn.Dropout(dropout),
                                        nn.Linear(H3, 1),
                                        nn.Linear(1, 1))
        self.init_weights()
        self.contextual_embedding = ContextualEmbedding(bert_model=bert_model, D_bert=D_bert, finetune=finetune_bert,
                                                        bert_pretrained_dir_path=bert_pretrained_dir_path)
        self.finetune_bert = finetune_bert

    def init_weights(self):
        w = (param.data for name, param in self.named_parameters() if 'weight' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        nn.init.uniform_(self.word_embeds.weight.data, a=-0.5, b=0.5)
        for t in w:
            nn.init.xavier_uniform_(t)
        for t in b:
            nn.init.constant_(t, 0)

    def load_embeddings(self, preloaded_weights):
        self.word_embeds.weight = nn.Parameter(preloaded_weights)

    def load_weights(self, weights_path):
        print("Loading weights")
        single_layers_weights, single_layers_biases = [], []
        for f in sorted(os.listdir(weights_path)):
            if f.startswith("single_mention_weights"):
                single_layers_weights.append(np.load(os.path.join(weights_path, f)))
            if f.startswith("single_mention_bias"):
                single_layers_biases.append(np.load(os.path.join(weights_path, f)))
        top_single_linear = (layer for layer in self.single_top if isinstance(layer, nn.Linear))
        for w, b, layer in zip(single_layers_weights, single_layers_biases, top_single_linear):
            layer.weight = nn.Parameter(torch.from_numpy(w).float())
            layer.bias = nn.Parameter(torch.from_numpy(b).float().squeeze())
        pair_layers_weights, pair_layers_biases = [], []
        for f in sorted(os.listdir(weights_path)):
            if f.startswith("pair_mentions_weights"):
                pair_layers_weights.append(np.load(os.path.join(weights_path, f)))
            if f.startswith("pair_mentions_bias"):
                pair_layers_biases.append(np.load(os.path.join(weights_path, f)))
        top_pair_linear = (layer for layer in self.pair_top if isinstance(layer, nn.Linear))
        for w, b, layer in zip(pair_layers_weights, pair_layers_biases, top_pair_linear):
            layer.weight = nn.Parameter(torch.from_numpy(w).float())
            layer.bias = nn.Parameter(torch.from_numpy(b).float().squeeze())

    def forward(self, inputs, concat_axis=1):
        inputs = tuple(inputs)
        pairs = (len(inputs) == 10)
        if pairs:
            spans, words, single_features, ant_spans, ant_words, ana_spans, ana_words, pair_features, contexts, masks \
             = inputs
        else:
            spans, words, single_features, contexts, masks = inputs

        embed_contexts = self.contextual_embedding(contexts, masks, self.finetune_bert)
        words = words.type(torch.LongTensor).to(self.device)
        embed_words = self.drop(self.word_embeds(words).view(words.size()[0], -1))
        single_input = torch.cat([spans, embed_words, single_features], 1)
        single_scores = self.single_top(single_input)
        if pairs:
            batchsize, pairs_num, _ = ana_spans.size()
            ant_words_long = ant_words.view(batchsize, -1).type(torch.LongTensor).to(self.device)
            ana_words_long = ana_words.view(batchsize, -1).type(torch.LongTensor).to(self.device)
            ant_embed_words = self.drop(self.word_embeds(ant_words_long).view(batchsize, pairs_num, -1))
            ana_embed_words = self.drop(self.word_embeds(ana_words_long).view(batchsize, pairs_num, -1))
            pair_input = torch.cat([ant_spans, ant_embed_words, ana_spans, ana_embed_words, pair_features], 2)
            pair_input = torch.cat([pair_input, embed_contexts])
            pair_scores = self.pair_top(pair_input).squeeze(dim=2)
            total_scores = torch.cat([pair_scores, single_scores], concat_axis)

        return total_scores if pairs else single_scores


class ContextualEmbedding(nn.Module):
    def __init__(self, bert_model='distilbert-base-uncased', D_bert=768, D_context_head=50,
                 finetune=False, bert_pretrained_dir_path=None):
        """
        Wrapper class for contextual embeddings by pooling BERT outputs
        :param bert_model: BERT model ID (huggingface transformers library or 'spanbert-base')
        :param D_bert: int, BERT token embedding dimensionality
        :param D_context_head: int, hidden state size for RNN
        :param finetune: boolean, whether to finetune the BERT model
        :param bert_pretrained_dir_path: Path to directory of pretrained BERT model weights (must be passed for SpanBERT)
        """
        super(ContextualEmbedding, self).__init__()
        if bert_model == 'distilbert-base-uncased':
            self.bert = DistilBertModel.from_pretrained(bert_model)
        elif bert_model == 'spanbert-base':
            if not bert_pretrained_dir_path:
                raise ValueError("Missing parameter bert_pretrained_dir_path for SpanBERT")
            self.bert = BertForSequenceClassification.from_pretrained(bert_pretrained_dir_path)
        else:
            raise ValueError("Unsupported BERT model: {}".format(bert_model))

        if finetune:
            print("Fine-tuning is not supported yet")
            pass  # TODO

        self.context_head = nn.GRU(D_bert, D_context_head, 1, bidirectional=True)

    def forward(self, inputs, masks, requires_grad=False):
        # Input shape: (bs, n_input_ids)
        if requires_grad:  # TODO
            # BERT output shape: (bs, n_input_ids, hidden_dim)
            embed_contexts = self.bert(input_ids=inputs, attention_mask=masks)[0]
        else:
            with torch.no_grad():
                # BERT output shape: (bs, n_input_ids, hidden_dim)
                embed_contexts = self.bert(input_ids=inputs, attention_mask=masks)[0]

        lengths = (masks == 1).sum(dim=1)
        packed = pack_padded_sequence(embed_contexts, lengths, batch_first=True)
        # Output shape: (bs, rnn_hidden_dim)
        return self.context_head(packed)[1].transpose(0, 1).flatten(1, 2)
