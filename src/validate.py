import torch 
import numpy as np
from torch import Tensor 
from typing import List 
import logging
import torch.nn.functional as F 
from src.model import Model
from src.data import UNK_ID, PAD_ID, BOS_ID, EOS_ID, BinaryData

logger = logging.getLogger(__name__)

def search(batch_data:BinaryData, model:Model, cfg:dict):
    """
    Get outputs and attention scores for a given batch.
    """
    mode = cfg["model"]["mode"]
    assembly_tokens = batch_data.assembly_tokens
    pseudo_tokens = batch_data.pseudo_tokens
    pseudo_tokens_codet5 = batch_data.pseudo_tokens_codet5
    cfg_nodes = batch_data.cfg_nodes
    cfg_edges = batch_data.cfg_edges
    node_batch = batch_data.batch
    # prt = batch_data.ptr
    assembly_token_mask = (assembly_tokens != PAD_ID).unsqueeze(1) # src_mask (batch, 1, code_token_length)
    # src_mask: normal is True; pad is False
    pseudo_token_mask = (pseudo_tokens != PAD_ID).unsqueeze(1)
    pseudo_tokens_codet5_mask = (pseudo_tokens_codet5 != 0).unsqueeze(1)

    with torch.no_grad():
        transformer_encoder_output, assembly_token_mask, gnn_encoder_output, node_mask, \
        pseudo_encoder_output, pseudo_token_mask = model(return_type="encode_{}".format(mode), 
                                                assembly_token_input=assembly_tokens, 
                                                cfg_node_input=cfg_nodes, 
                                                cfg_node_batch=node_batch,
                                                edge_index=cfg_edges, 
                                                trg_input=None, 
                                                trg_truth=None, 
                                                assembly_token_mask=assembly_token_mask, 
                                                cfg_node_mask=None,
                                                trg_mask=None,
                                                transformer_encoder_output=None,
                                                gnn_encoder_output=None,
                                                pseudo_encoder_output=None,
                                                pseudo_token_input=pseudo_tokens,
                                                pseudo_token_mask=pseudo_token_mask,
                                                pseudo_token_codet5_input=pseudo_tokens_codet5,
                                                pseudo_token_codet5_mask=pseudo_tokens_codet5_mask)
        
    beam_size = cfg["testing"].get("beam_size", 4)
    beam_alpha = cfg["testing"].get("beam_alpha", -1)
    max_output_length = cfg["testing"].get("max_output_length", 40)
    min_output_length = cfg["testing"].get("min_output_length", 1)
    n_best = cfg["testing"].get("n_best", 1)
    return_attention = cfg["testing"].get("return_attention", True)
    return_probability = cfg["testing"].get("return_probability", True) 
    generate_unk = cfg["testing"].get("generate_unk", False)
    # repetition_penalty = cfg["testing"].get("repetition_penalty", -1)

    if beam_size < 2: 
        stacked_output, stacked_probability, stacked_attention = greedy_search(model=model, 
                                                                               transformer_encoder_output=transformer_encoder_output,
                                                                               src_mask=assembly_token_mask,
                                                                               gnn_encoder_output=gnn_encoder_output,
                                                                               node_mask=node_mask,
                                                                               pseudo_encoder_output=pseudo_encoder_output,
                                                                               pseudo_token_mask=pseudo_token_mask,
                                                                               max_output_length=max_output_length,
                                                                               min_output_length=min_output_length,
                                                                               generate_unk=generate_unk,
                                                                               return_attention=return_attention,
                                                                               return_probability=return_probability,
                                                                               mode=mode)
    else:
        stacked_output, stacked_probability, stacked_attention = beam_search(model=model, 
                                                                             transformer_encoder_output=transformer_encoder_output,
                                                                             src_mask=assembly_token_mask, 
                                                                             gnn_encoder_output=gnn_encoder_output,
                                                                             node_mask=node_mask,
                                                                             pseudo_encoder_output=pseudo_encoder_output,
                                                                             pseudo_token_mask=pseudo_token_mask,
                                                                             max_output_length=max_output_length,
                                                                             min_output_length=min_output_length,
                                                                             beam_size=beam_size,
                                                                             beam_alpha=beam_alpha,
                                                                             n_best=n_best,
                                                                             generate_unk=generate_unk,
                                                                             return_attention=return_attention,
                                                                             return_probability=return_probability,
                                                                             mode=mode)
        
    return stacked_output, stacked_probability, stacked_attention

def greedy_search(model, transformer_encoder_output, src_mask, gnn_encoder_output, node_mask, pseudo_encoder_output, pseudo_token_mask,
                  max_output_length, min_output_length, generate_unk, return_attention, return_probability, mode):
    """
    Transformer Greedy function.
    :param: model: Transformer Model
    :param: encoder_output: [batch_size, src_len, model_dim]
    :param: src_mask: [batch_size, 1, src_len] # src_len is padded src length
    return
        - stacked_output [batch_size, steps/max_output_length]
        - stacked_scores [batch_size, steps/max_output_length] # log_softmax token probability
        - stacked_attention [batch_size, steps/max_output_length, src_len]
    """
    unk_index = UNK_ID
    pad_index = PAD_ID
    bos_index = BOS_ID
    eos_index = EOS_ID

    batch_size, _, src_length = src_mask.size()

    # start with BOS-symbol for each sentence in the batch
    if transformer_encoder_output is not None:
        generated_tokens = transformer_encoder_output.new_full((batch_size,1), bos_index, dtype=torch.long, requires_grad=False)
    else:
        generated_tokens = pseudo_encoder_output.new_full((batch_size,1), bos_index, dtype=torch.long, requires_grad=False)
    # generated_tokens [batch_size, 1] generated_tokens id

    # Placeholder for scores
    generated_scores = generated_tokens.new_zeros((batch_size,1), dtype=torch.float) if return_probability is True else None
    # generated_scores [batch_size, 1]

    # Placeholder for attentions
    generated_attention_weight = generated_tokens.new_zeros((batch_size, 1, src_length), dtype=torch.float) if return_attention else None
    # generated_attention_weight [batch_size, 1, src_len]

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones((1, 1, 1))

    finished = src_mask.new_zeros(batch_size).byte() # [batch_size], uint8

    for step in range(max_output_length):
        with torch.no_grad():
            logits = model(return_type="decode_{}".format(mode), 
                            assembly_token_input=None, 
                            cfg_node_input=None, 
                            cfg_node_batch=None,
                            edge_index=None, 
                            trg_input=generated_tokens, 
                            trg_truth=None, 
                            assembly_token_mask=src_mask, 
                            cfg_node_mask=node_mask, 
                            trg_mask=trg_mask, 
                            transformer_encoder_output=transformer_encoder_output,
                            gnn_encoder_output=gnn_encoder_output,
                            pseudo_encoder_output=pseudo_encoder_output,
                            pseudo_token_input=None,
                            pseudo_token_mask=pseudo_token_mask)
            # logits [batch_size, step+1, trg_vocab_size]

            output = logits[:, -1] 
            # output [batch_size, trg_vocab_size]
            if not generate_unk:
                output[:, unk_index] = float("-inf")
            if step < min_output_length:
                output[:, eos_index] = float("-inf")
            output = F.softmax(output, dim=-1)

        # take the most likely token
        prob, next_words = torch.max(output, dim=-1)
        # prob [batch_size]
        # next_words [batch_size]

        generated_tokens = torch.cat([generated_tokens, next_words.unsqueeze(-1)], dim=-1) # [batch_size, step+2]

        if return_attention is True:
            generated_scores = torch.cat([generated_scores, prob.unsqueeze(-1)], dim=-1) # [batch_size, step+2]

        if return_attention is True:
            # cross_attention = cross_attention_weight_gnn.data[:, -1, :].unsqueeze(1) # [batch_size, 1, src_len]
            cross_attention=None
            generated_attention_weight = torch.cat([generated_attention_weight, cross_attention], dim=1) # [batch_size, step+2, src_len]
    
        # check if previous symbol is <eos>
        is_eos = torch.eq(next_words, eos_index)
        finished += is_eos
        if (finished >= 1).sum() == batch_size:
            break

    # Remove bos-symbol
    # FIXME why need to cpu
    stacked_output = generated_tokens[:, 1:].detach().cpu().long()
    stacked_probability = generated_scores[:, 1:].detach().cpu().float() if return_probability  else None
    stacked_attention = generated_attention_weight[:, 1:, :].detach().cpu().float() if return_attention else None
    return stacked_output, stacked_probability, stacked_attention

def tile(x: Tensor, count: int, dim : int=0) -> Tensor:
    """
    Tiles x on dimension 'dim' count times. Used for beam search.
    i.e. [a,b] --count=3--> [a,a,a,b,b,b]
    :param: x [batch_size, src_len, model_dim]
    return tiled tensor
    """
    assert dim == 0
    out_size = list(x.size()) # [batch_size, src_len, model_dim]
    out_size[0] = out_size[0] * count # [batch_size*count, src_len, model_dim]
    batch_size = x.size(0)
    x = x.view(batch_size, -1).transpose(0,1).repeat(count, 1).transpose(0,1).contiguous().view(*out_size)
    return x

def beam_search(model, transformer_encoder_output, src_mask, gnn_encoder_output, node_mask, pseudo_encoder_output, pseudo_token_mask,
                max_output_length, min_output_length, beam_size, beam_alpha, n_best, generate_unk, return_attention, return_probability, mode):
    """
    Transformer Beam Search function.
    In each decoding step, find the k most likely partial hypotheses.
    Inspired by OpenNMT-py, adapted for Transformer.
    :param: model: Transformer Model
    :param: encoder_output: [batch_size, src_len, model_dim]
    :param: src_mask: [batch_size, 1, src_len] # src_len is padded src length
    return
        - final_output [batch_size*n_best, hyp_len]
        - scores
        - attention: None 
    """
    assert beam_size > 0, "Beam size must be > 0."
    assert n_best <= beam_size, f"Can only return {beam_size} best hypotheses."

    unk_index = UNK_ID
    pad_index = PAD_ID
    bos_index = BOS_ID
    eos_index = EOS_ID
    batch_size = src_mask.size(0)

    trg_vocab_size = model.vocab_info["comment_token_vocab"]["size"]
    trg_mask = None 
    if transformer_encoder_output is not None:
        device = transformer_encoder_output.device
    else:
        device = pseudo_encoder_output.device

    if transformer_encoder_output is not None:
        transformer_encoder_output = tile(transformer_encoder_output.contiguous(), beam_size, dim=0)
    else:
        transformer_encoder_output = None  
    # encoder_output [batch_size*beam_size, src_len, model_dim] i.e. [a,a,a,b,b,b]

    src_mask = tile(src_mask, beam_size, dim=0)
    # src_mask [batch_size*beam_size, 1, src_len]
    
    if mode == "assembly_cfg_comment" or mode == "assembly_cfg_pseudo_comment" or mode == "assembly_cfg_pseudo_codet5_comment":
        gnn_encoder_output = tile(gnn_encoder_output.contiguous(), beam_size, dim=0)
        node_mask = tile(node_mask, beam_size, dim=0)
    elif mode == "assembly_comment" or mode == "assembly_pseudo_comment" or mode == "pseudo_comment":
        gnn_encoder_output = None 
        node_mask= None 
    else:
        assert False, "mode name error."
    
    if mode == "assembly_cfg_pseudo_comment" or mode == "assembly_pseudo_comment" or mode == "pseudo_comment" or mode == "assembly_cfg_pseudo_codet5_comment":
        pseudo_encoder_output = tile(pseudo_encoder_output.contiguous(), beam_size, dim=0)
        pseudo_token_mask = tile(pseudo_token_mask, beam_size, dim=0)
    else:
        pseudo_encoder_output = None
        pseudo_token_mask = None

    trg_mask = src_mask.new_ones((1,1,1))

    batch_offset = torch.arange(batch_size, dtype=torch.long, device=device) # [0,1,2,... batch_size-1]
    beam_offset = torch.arange(0, batch_size*beam_size, step=beam_size, dtype=torch.long, device=device)
    # beam_offset [0,5,10,15,....] i.e. beam_size=5

    # keep track of the top beam size hypotheses to expand for each element
    # in the batch to be futher decoded (that are still "alive")
    alive_sentences = torch.full((batch_size*beam_size, 1), bos_index, dtype=torch.long, device=device)
    # alive_sentences [batch_size*beam_size, hyp_len] now is [batch_size*beam_size, 1]

    top_k_log_probs = torch.zeros(batch_size, beam_size, device=device)
    top_k_log_probs[:, 1:] = float("-inf")

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {"predictions": [[] for _ in range(batch_size)], 
                "scores": [[] for _ in range(batch_size)] }

    # Indicator if the generation is finished.
    is_finished = torch.full((batch_size, beam_size), False, dtype=torch.bool, device=device)

    for step in range(max_output_length):
        # feed the complete predicted sentences so far.
        decoder_input = alive_sentences
        with torch.no_grad():
            logits = model(return_type="decode_{}".format(mode), 
                            assembly_token_input=None, 
                            cfg_node_input=None, 
                            cfg_node_batch=None,
                            edge_index=None, 
                            trg_input=decoder_input,
                            trg_truth=None, 
                            assembly_token_mask=src_mask, 
                            cfg_node_mask=node_mask, 
                            trg_mask=trg_mask,
                            transformer_encoder_output=transformer_encoder_output,
                            gnn_encoder_output=gnn_encoder_output,
                            pseudo_encoder_output=pseudo_encoder_output,
                            pseudo_token_input=None,
                            pseudo_token_mask=pseudo_token_mask)

            # for the transformer we made predictions for all time steps up to this point, so we only want to know about the last time step.
            output = logits[:, -1] # output [batch_size*beam_size, vocab_size]

        # compute log probability distribution over trg vocab
        log_probs = F.log_softmax(output, dim=-1)
        # log_probs [batch_size*beam_size, vocab_size]

        if not generate_unk:
            log_probs[:, unk_index] = float("-inf")
        if step < min_output_length:
            log_probs[:, eos_index] = float("-inf")

        # multiply probs by the beam probability (means add log_probs after log operation)
        log_probs += top_k_log_probs.view(-1).unsqueeze(1)
        current_scores = log_probs.clone()

        # compute length penalty
        if beam_alpha > 0:
            length_penalty = ((5.0 + (step+1)) / 6.0)**beam_alpha
            current_scores /= length_penalty
        
        # flatten log_probs into a list of possibilities
        current_scores = current_scores.reshape(-1, beam_size*trg_vocab_size)
        # current_scores [batch_size, beam_size*vocab_size]

        # pick currently best top k hypotheses
        topk_scores, topk_ids =current_scores.topk(beam_size, dim=-1)
        # topk_scores [batch_size, beam_size]
        # topk_ids [batch_size, beam_size]

        if beam_alpha > 0:
            top_k_log_probs = topk_scores * length_penalty
        else: 
            top_k_log_probs = topk_scores.clone()
        
        # Reconstruct beam origin and true word ids from flatten order
        topk_beam_index = topk_ids.div(trg_vocab_size, rounding_mode="floor")
        # topk_beam_index [batch_size, beam_size]
        topk_ids = topk_ids.fmod(trg_vocab_size) # true word ids
        # topk_ids [batch_size, beam_size]

        # map topk_beam_index to batch_index in the flat representation
        batch_index = topk_beam_index + beam_offset[:topk_ids.size(0)].unsqueeze(1)
        # batch_index [batch_size, beam_size]
        select_indices = batch_index.view(-1)
        # select_indices [batch_size*beam_size]: the number of seleced index in the batch.

        # append latest prediction
        alive_sentences = torch.cat([alive_sentences.index_select(0, select_indices), topk_ids.view(-1, 1)], dim=-1)
        # alive_sentences [batch_size*beam_size, hyp_len]

        is_finished = topk_ids.eq(eos_index) | is_finished | topk_scores.eq(-np.inf)
        # is_finished [batch_size, beam_size]
        if step + 1 == max_output_length:
            is_finished.fill_(True)
        
        # end condition is whether all beam candidates in each example are finished.
        end_condition = is_finished.all(dim=-1)
        # end_condition [batch_size]

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_sentences.view(-1, beam_size, alive_sentences.size(-1))
            # predictions [batch_size, beam_size, hyp_len]

            for sentence_idx in range(is_finished.size(0)): # look over sentences
                b = batch_offset[sentence_idx].item() # index of that example in the batch
                if end_condition[sentence_idx]:
                    is_finished[sentence_idx].fill_(True)
                
                finished_hyp = is_finished[sentence_idx].nonzero(as_tuple=False).view(-1)
                for sentence_beam_idx in finished_hyp: # look over finished beam candidates
                    number_eos = (predictions[sentence_idx, sentence_beam_idx, 1:] == eos_index).count_nonzero().item()
                    if number_eos > 1: # prediction should have already been added to the hypotheses
                        continue
                    elif (number_eos == 0 and step+1 == max_output_length) or (number_eos == 1 and predictions[sentence_idx, sentence_beam_idx, -1] == eos_index):
                        hypotheses[b].append((topk_scores[sentence_idx, sentence_beam_idx], predictions[sentence_idx, sentence_beam_idx,1:]))

                # if all n best candidates of the i-the example reached the end, save them
                if end_condition[sentence_idx]:
                    best_hyp = sorted(hypotheses[b], key=lambda x:x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break 
                        if len(pred) < max_output_length:
                            assert pred[-1] == eos_index, "Add a candidate which doesn't end with eos."
                        
                        results['scores'][b].append(score)
                        results['predictions'][b].append(pred)
            
            # batch indices of the examples which contain unfinished candidates.
            unfinished = end_condition.eq(False).nonzero(as_tuple=False).view(-1)
            # unfinished [batch_size]
            if len(unfinished) == 0:
                break
            
            # remove finished examples for the next steps.
            # shape [remaining_batch_size, beam_size]
            batch_index = batch_index.index_select(0, unfinished)
            top_k_log_probs = top_k_log_probs.index_select(0, unfinished)
            is_finished = is_finished.index_select(0, unfinished)
            batch_offset = batch_offset.index_select(0, unfinished)

            alive_sentences = predictions.index_select(0, unfinished).view(-1, alive_sentences.size(-1))

        # Reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        if transformer_encoder_output is not None:
            transformer_encoder_output = transformer_encoder_output.index_select(0, select_indices)
        else:
            transformer_encoder_output = None 
        src_mask = src_mask.index_select(0, select_indices)
        if mode == "assembly_cfg_comment" or mode == "assembly_cfg_pseudo_comment" or mode == "assembly_cfg_pseudo_codet5_comment":
            gnn_encoder_output = gnn_encoder_output.index_select(0, select_indices)
            node_mask = node_mask.index_select(0, select_indices)
        else: # mode = "assembly_comment"
            gnn_encoder_output = None
            node_mask = None 
        
        if mode == "assembly_cfg_pseudo_comment" or mode == "assembly_pseudo_comment" or mode == "pseudo_comment" or  mode == "assembly_cfg_pseudo_codet5_comment":
            pseudo_encoder_output = pseudo_encoder_output.index_select(0, select_indices)
            pseudo_token_mask = pseudo_token_mask.index_select(0, select_indices)
        else:
            pseudo_encoder_output = None 
            pseudo_token_mask = None 

    def pad_and_stack_hyps(hyps: List[np.ndarray]):
        filled = (np.ones((len(hyps), max([h.shape[0]  for h in hyps])), dtype=int) * pad_index)
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked output
    # final_outputs [batch_size*n_best, hyp_len]
    predictions_list = [u.cpu().numpy() for r in results["predictions"] for u in r]
    final_outputs = pad_and_stack_hyps(predictions_list)
    scores = (np.array([[u.item()] for r in results['scores'] for u in r]) if return_probability else None)

    return final_outputs, scores, None


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
    search()