import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import  scatter_max
import re 
import json 

def test1():
    class SmallNet(torch.nn.Module):
        def __init__(self):
            super(SmallNet, self).__init__()
            self.conv1 = GCNConv(2, 4)
            self.linear1 = torch.nn.Linear(4,3)

        def forward(self, data):
            x, edge_index, token = data.node, data.node_edge, data.token
            print("x = {}".format(x))
            print("edge = {}".format(edge_index))
            print("token = {} shape = {}".format(token, token.shape))
            assert False
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x, _ = scatter_max(x, data.batch, dim=0)
            x = self.linear1(x)
            return x

    class Datap(Data):
        def __init__(self, node=None, node_edge=None, token=None, label=None):
            super().__init__()
            self.node = node
            self.token = token
            self.node_edge = node_edge
            self.label = label 
            # self.__num_nodes__ = node.size(0)
        
        # 自定义拼接步长
        def __inc__(self, key, value, *args, **kwargs):  # 增量，针对edge_index
            if key == 'node_edge':
                return self.node.size(0)
            else:
                return 0

        def __cat_dim__(self, key, value, *args, **kwargs):
            if bool(re.search('node_edge', key)):
                return -1
            elif bool(re.search('node',key)):
                return 0
            elif bool(re.search('token', key)):
                return None 
            else:
                super().__cat_dim__(key, value, *args, **kwargs)


    def init_data():
        labels=np.array([0,1,2],dtype=int)
        a=labels[0]
        data_list = []
        
        #定义第一个节点的信息
        x = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])
        x = torch.tensor(x, dtype=torch.float)
        edge = np.array([
            [0, 0, 2],
            [1, 2, 0]
        ])
        edge = torch.tensor(edge, dtype=torch.long)
        token = torch.tensor([11,12,13,14], dtype=torch.long)
        data_list.append(Datap(node=x, node_edge=edge.contiguous(), token=token, label=int(labels[0])))

        #定义第二个节点的信息
        x = np.array([
            [0, 0],
            [1, 1],
            [2, 2]
        ])
        x = torch.tensor(x, dtype=torch.float)
        edge = np.array([
            [0, 1],
            [1, 2]
        ])
        edge = torch.tensor(edge, dtype=torch.long)
        token = torch.tensor([21,22,23,24], dtype=torch.long)
        data_list.append(Datap(node=x, node_edge=edge.contiguous(), token=token,label=int(labels[1])))

        #定义第三个节点的信息
        x = np.array([
            [0, 0],
        [1, 1],
            [2, 2]
        ])
        x = torch.tensor(x, dtype=torch.float)
        edge = np.array([
            [0, 1, 2],
            [2, 2, 0]
        ])
        edge = torch.tensor(edge, dtype=torch.long)
        token = torch.tensor([31,32,33,34], dtype=torch.long)
        data_list.append(Datap(node=x, node_edge=edge.contiguous(), token=token, label=int(labels[2])))
        return data_list

    epoch_num=10000
    batch_size=2
    trainset=init_data()
    #NOTE 
    item0 = trainset[0]
    print(item0)
    print('edge_attr' in item0)
    print(item0.num_nodes)
    print(item0.num_node_features)
    print("is directed = {}".format(item0.is_direceted()))
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=False)

    device = torch.device('cpu')
    model = SmallNet().to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epoch_num):
        train_loss = 0.0
        for i, batch in enumerate(trainloader):
            #print("label = {}".format(batch.label))
            batch = batch.to("cpu")
            optimizer.zero_grad()
            outputs = model(batch)
            #print(outputs)
            #print(batch.label)
            loss = criterion(outputs, batch.label)
            #print("loss = {}".format(loss))
            loss.backward()
            optimizer.step()

            train_loss += loss.cpu().item()
            # print('epoch: {:d} loss: {:.3f}'
            #       .format(epoch + 1, loss.cpu().item()))
        print('epoch: {:d} loss: {:.3f}'
            .format(epoch + 1, train_loss / batch_size))

def test2():
    scores = torch.tensor([1,2,3,4])
    mask = torch.BoolTensor([1,1,0,0])
    scores = scores.masked_fill(~mask, value=-100)
    print(scores)
    return None

def test3():
    size = 4
    ones = torch.ones(size,size, dtype=torch.bool)
    subsequence_mask = torch.tril(ones, out=ones).unsqueeze(0)
    print(subsequence_mask)

    trg_mask = torch.tensor([[1,1,1,0],[1,0,0,0],[1,1,0,0]]).view(3,1,4)

    mask = subsequence_mask & trg_mask
    print(mask)


from torch_geometric.utils import to_dense_batch
def test4():
    x = torch.arange(18).view(6,3)
    print("x={}".format(x))

    batch = torch.tensor([0,0,1,2,2,2])
    out, mask = to_dense_batch(x, batch, max_num_nodes=3)
    print("out = {}".format(out))
    print("mask = {}".format(mask))


def test_result():
    our_result = "test_best-bleu.output"
    truth = "../datas/dataset_gcc-7.3.0_x86_64_O1_strip/test.json"
    with open(our_result, 'r') as paper, open(truth, 'r') as gold:
        fpaper = paper.read().splitlines()
        fpaper_get = []
        for item in fpaper:
            item_get = item.split('\t')[1]
            fpaper_get.append(item_get)

        # fgold = gold.read().splitlines()
        fgold_get = []
        fgold = json.load(gold)
        for item in fgold:
            comment = item["comment"]
            fgold_get.append(comment)

        bleu, rouge_l, meteor = eval_accuracies(fpaper_get, fgold_get)
        print("bleu = {}".format(bleu))
        print("rouge-l = {}".format(rouge_l))
        print("meteor = {}".format(meteor))
    return None 


def eval_accuracies(model_generated, target_truth):
    generated = {k: [v.strip()] for k, v in enumerate(model_generated)}
    target_truth = {k: [v.strip()] for k, v in enumerate(target_truth)}
    assert sorted(generated.keys()) == sorted(target_truth.keys())

    # Compute BLEU scores
    corpus_bleu_r, bleu, ind_bleu, bleu_4 = corpus_bleu(generated, target_truth)

    # Compute ROUGE_L scores
    rouge_calculator = Rouge()
    rouge_l, ind_rouge = rouge_calculator.compute_score(target_truth, generated)

    # Compute METEOR scores
    meteor_calculator = Meteor()
    meteor, _ = meteor_calculator.compute_score(target_truth, generated)


    return bleu * 100, rouge_l * 100, meteor * 100


# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Python implementation of BLEU and smooth-BLEU.
This module provides a Python implementation of BLEU and smooth-BLEU.
Smooth BLEU is computed following the method outlined in the paper:
Chin-Yew Lin, Franz Josef Och. ORANGE: a method for evaluating automatic
evaluation metrics for machine translation. COLING 2004.
"""

import collections
import math


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
      segment: text segment from which n-grams will be extracted.
      max_order: maximum length in tokens of the n-grams returned by this
          methods.
    Returns:
      The Counter containing all n-grams upto max_order in segment
      with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
      reference_corpus: list of lists of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
      3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
      precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus,
                                         translation_corpus):
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram) - 1] += overlap[ngram]
        for order in range(1, max_order + 1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order - 1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def corpus_bleu(hypotheses, references):
    refs = []
    hyps = []
    count = 0
    total_score = 0.0

    assert (sorted(hypotheses.keys()) == sorted(references.keys()))
    Ids = list(hypotheses.keys())
    ind_score = dict()

    for id in Ids:
        hyp = hypotheses[id][0].split()
        ref = [r.split() for r in references[id]]
        hyps.append(hyp)
        refs.append(ref)

        score = compute_bleu([ref], [hyp], smooth=True)[0]
        total_score += score
        count += 1
        ind_score[id] = score

    avg_score = total_score / count
    corpus_bleu = compute_bleu(refs, hyps, smooth=True)[0]
    bleu_4 = compute_bleu(refs, hyps, smooth=True)[1][3] * compute_bleu(refs, hyps, smooth=True)[2]
    return corpus_bleu, avg_score, ind_score, bleu_4



#!/usr/bin/env python
#
# File Name : rouge.py
#
# Description : Computes ROUGE-L metric as described by Lin and Hovey (2004)
#
# Creation Date : 2015-01-07 06:03
# Author : Ramakrishna Vedantam <vrama91@vt.edu>

import numpy as np


def my_lcs(string, sub):
    """
    Calculates longest common subsequence for a pair of tokenized strings
    :param string : list of str : tokens from a string split using whitespace
    :param sub : list of str : shorter string, also split using whitespace
    :returns: length (list of int): length of the longest common subsequence between the two strings
    Note: my_lcs only gives length of the longest common subsequence, not the actual LCS
    """
    if len(string) < len(sub):
        sub, string = string, sub

    lengths = [[0 for i in range(0, len(sub) + 1)] for j in range(0, len(string) + 1)]

    for j in range(1, len(sub) + 1):
        for i in range(1, len(string) + 1):
            if string[i - 1] == sub[j - 1]:
                lengths[i][j] = lengths[i - 1][j - 1] + 1
            else:
                lengths[i][j] = max(lengths[i - 1][j], lengths[i][j - 1])

    return lengths[len(string)][len(sub)]


class Rouge():
    '''
    Class for computing ROUGE-L score for a set of candidate sentences for the MS COCO test set
    '''

    def __init__(self):
        # vrama91: updated the value below based on discussion with Hovey
        self.beta = 1.2

    def calc_score(self, candidate, refs):
        """
        Compute ROUGE-L score given one candidate and references for an image
        :param candidate: str : candidate sentence to be evaluated
        :param refs: list of str : COCO reference sentences for the particular image to be evaluated
        :returns score: int (ROUGE-L score for the candidate evaluated against references)
        """
        assert (len(candidate) == 1)
        assert (len(refs) > 0)
        prec = []
        rec = []

        # split into tokens
        token_c = candidate[0].split(" ")

        for reference in refs:
            # split into tokens
            token_r = reference.split(" ")
            # compute the longest common subsequence
            lcs = my_lcs(token_r, token_c)
            prec.append(lcs / float(len(token_c)))
            rec.append(lcs / float(len(token_r)))

        prec_max = max(prec)
        rec_max = max(rec)

        if prec_max != 0 and rec_max != 0:
            score = ((1 + self.beta ** 2) * prec_max * rec_max) / float(rec_max + self.beta ** 2 * prec_max)
        else:
            score = 0.0
        return score

    def compute_score(self, gts, res):
        """
        Computes Rouge-L score given a set of reference and candidate sentences for the dataset
        Invoked by evaluate_captions.py
        :param gts: dict : candidate / test sentences with "image name" key and "tokenized sentences" as values
        :param res: dict : reference MS-COCO sentences with "image name" key and "tokenized sentences" as values
        :returns: average_score: float (mean ROUGE-L score computed by averaging scores for all the images)
        """
        assert (sorted(gts.keys()) == sorted(res.keys()))
        imgIds = list(gts.keys())

        score = dict()
        for id in imgIds:
            hypo = res[id]
            ref = gts[id]

            # Sanity check.
            assert (type(hypo) is list)
            assert (len(hypo) == 1)
            assert (type(ref) is list)
            assert (len(ref) > 0)

            score[id] = self.calc_score(hypo, ref)

        average_score = np.mean(np.array(list(score.values())))
        return average_score, score

    def method(self):
        return "Rouge"



#!/usr/bin/env python

# Python wrapper for METEOR implementation, by Xinlei Chen
# Acknowledge Michael Denkowski for the generous discussion and help
# from __future__ import division

import atexit
import logging
import os
import subprocess
import sys
import threading

import psutil

# Assumes meteor-1.5.jar is in the same directory as meteor.py.  Change as needed.
METEOR_JAR = 'meteor-1.5.jar'


def enc(s):
    return s.encode('utf-8')


def dec(s):
    return s.decode('utf-8')


class Meteor:

    def __init__(self):
        # Used to guarantee thread safety
        self.lock = threading.Lock()

        mem = '2G'
        mem_available_G = psutil.virtual_memory().available / 1E9
        if mem_available_G < 2:
            logging.warning("There is less than 2GB of available memory.\n"
                            "Will try with limiting Meteor to 1GB of memory but this might cause issues.\n"
                            "If you have problems using Meteor, "
                            "then you can try to lower the `mem` variable in meteor.py")
            mem = '1G'

        meteor_cmd = ['java', '-jar', '-Xmx{}'.format(mem), METEOR_JAR,
                      '-', '-', '-stdio', '-l', 'en', '-norm']
        env = os.environ.copy()
        env['LC_ALL'] = "C"
        self.meteor_p = subprocess.Popen(meteor_cmd,
                                         cwd=os.path.dirname(os.path.abspath(__file__)),
                                         env=env,
                                         stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.PIPE)

        atexit.register(self.close)

    def close(self):
        with self.lock:
            if self.meteor_p:
                self.meteor_p.kill()
                self.meteor_p.wait()
                self.meteor_p = None
        # if the user calls close() manually, remove the
        # reference from atexit so the object can be garbage-collected.
        if atexit is not None and atexit.unregister is not None:
            atexit.unregister(self.close)

    def compute_score(self, gts, res):
        assert (gts.keys() == res.keys())
        imgIds = gts.keys()
        scores = []

        eval_line = 'EVAL'
        with self.lock:
            for i in imgIds:
                assert (len(res[i]) == 1)
                stat = self._stat(res[i][0], gts[i])
                eval_line += ' ||| {}'.format(stat)

            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            for i in range(0, len(imgIds)):
                v = self.meteor_p.stdout.readline()
                try:
                    scores.append(float(dec(v.strip())))
                except:
                    sys.stderr.write("Error handling value: {}\n".format(v))
                    sys.stderr.write("Decoded value: {}\n".format(dec(v.strip())))
                    sys.stderr.write("eval_line: {}\n".format(eval_line))
                    # You can try uncommenting the next code line to show stderr from the Meteor JAR.
                    # If the Meteor JAR is not writing to stderr, then the line will just hang.
                    # sys.stderr.write("Error from Meteor:\n{}".format(self.meteor_p.stderr.read()))
                    raise
            score = float(dec(self.meteor_p.stdout.readline()).strip())

        return score, scores

    def method(self):
        return "METEOR"

    def _stat(self, hypothesis_str, reference_list):
        # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
        hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
        score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
        self.meteor_p.stdin.write(enc(score_line))
        self.meteor_p.stdin.write(enc('\n'))
        self.meteor_p.stdin.flush()
        return dec(self.meteor_p.stdout.readline()).strip()

    def _score(self, hypothesis_str, reference_list):
        with self.lock:
            # SCORE ||| reference 1 words ||| reference n words ||| hypothesis words
            hypothesis_str = hypothesis_str.replace('|||', '').replace('  ', ' ')
            score_line = ' ||| '.join(('SCORE', ' ||| '.join(reference_list), hypothesis_str))
            self.meteor_p.stdin.write(enc('{}\n'.format(score_line)))
            self.meteor_p.stdin.flush()
            stats = dec(self.meteor_p.stdout.readline()).strip()
            eval_line = 'EVAL ||| {}'.format(stats)
            # EVAL ||| stats 
            self.meteor_p.stdin.write(enc('{}\n'.format(eval_line)))
            self.meteor_p.stdin.flush()
            score = float(dec(self.meteor_p.stdout.readline()).strip())
            # bug fix: there are two values returned by the jar file, one average, and one all, so do it twice
            # thanks for Andrej for pointing this out
            score = float(dec(self.meteor_p.stdout.readline()).strip())
        return score

    def __del__(self):
        self.close()


if __name__ == "__main__":
    test_result()
    # test6()