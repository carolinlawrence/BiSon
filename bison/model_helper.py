# coding=utf-8
#        BiSon
#
#   File:     model_helper.py
#   Authors:  Carolin Lawrence carolin.lawrence@neclab.eu
#             Bhushan Kotnis bhushan.kotnis@neclab.eu
#             Mathias Niepert mathias.niepert@neclab.eu
#
# NEC Laboratories Europe GmbH, Copyright (c) 2019, All rights reserved.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
#
#        PROPRIETARY INFORMATION ---
#
# SOFTWARE LICENSE AGREEMENT
#
# ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
#
# BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS
# LICENSE AGREEMENT.  IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR
# DOWNLOAD THE SOFTWARE.
#
# This is a license agreement ("Agreement") between your academic institution
# or non-profit organization or self (called "Licensee" or "You" in this
# Agreement) and NEC Laboratories Europe GmbH (called "Licensor" in this
# Agreement).  All rights not specifically granted to you in this Agreement
# are reserved for Licensor.
#
# RESERVATION OF OWNERSHIP AND GRANT OF LICENSE: Licensor retains exclusive
# ownership of any copy of the Software (as defined below) licensed under this
# Agreement and hereby grants to Licensee a personal, non-exclusive,
# non-transferable license to use the Software for noncommercial research
# purposes, without the right to sublicense, pursuant to the terms and
# conditions of this Agreement. NO EXPRESS OR IMPLIED LICENSES TO ANY OF
# LICENSOR'S PATENT RIGHTS ARE GRANTED BY THIS LICENSE. As used in this
# Agreement, the term "Software" means (i) the actual copy of all or any
# portion of code for program routines made accessible to Licensee by Licensor
# pursuant to this Agreement, inclusive of backups, updates, and/or merged
# copies permitted hereunder or subsequently supplied by Licensor,  including
# all or any file structures, programming instructions, user interfaces and
# screen formats and sequences as well as any and all documentation and
# instructions related to it, and (ii) all or any derivatives and/or
# modifications created or made by You to any of the items specified in (i).
#
# CONFIDENTIALITY/PUBLICATIONS: Licensee acknowledges that the Software is
# proprietary to Licensor, and as such, Licensee agrees to receive all such
# materials and to use the Software only in accordance with the terms of this
# Agreement.  Licensee agrees to use reasonable effort to protect the Software
# from unauthorized use, reproduction, distribution, or publication. All
# publication materials mentioning features or use of this software must
# explicitly include an acknowledgement the software was developed by NEC
# Laboratories Europe GmbH.
#
# COPYRIGHT: The Software is owned by Licensor.
#
# PERMITTED USES:  The Software may be used for your own noncommercial
# internal research purposes. You understand and agree that Licensor is not
# obligated to implement any suggestions and/or feedback you might provide
# regarding the Software, but to the extent Licensor does so, you are not
# entitled to any compensation related thereto.
#
# DERIVATIVES: You may create derivatives of or make modifications to the
# Software, however, You agree that all and any such derivatives and
# modifications will be owned by Licensor and become a part of the Software
# licensed to You under this Agreement.  You may only use such derivatives and
# modifications for your own noncommercial internal research purposes, and you
# may not otherwise use, distribute or copy such derivatives and modifications
# in violation of this Agreement.
#
# BACKUPS:  If Licensee is an organization, it may make that number of copies
# of the Software necessary for internal noncommercial use at a single site
# within its organization provided that all information appearing in or on the
# original labels, including the copyright and trademark notices are copied
# onto the labels of the copies.
#
# USES NOT PERMITTED:  You may not distribute, copy or use the Software except
# as explicitly permitted herein. Licensee has not been granted any trademark
# license as part of this Agreement.  Neither the name of NEC Laboratories
# Europe GmbH nor the names of its contributors may be used to endorse or
# promote products derived from this Software without specific prior written
# permission.
#
# You may not sell, rent, lease, sublicense, lend, time-share or transfer, in
# whole or in part, or provide third parties access to prior or present
# versions (or any parts thereof) of the Software.
#
# ASSIGNMENT: You may not assign this Agreement or your rights hereunder
# without the prior written consent of Licensor. Any attempted assignment
# without such consent shall be null and void.
#
# TERM: The term of the license granted by this Agreement is from Licensee's
# acceptance of this Agreement by downloading the Software or by using the
# Software until terminated as provided below.
#
# The Agreement automatically terminates without notice if you fail to comply
# with any provision of this Agreement.  Licensee may terminate this Agreement
# by ceasing using the Software.  Upon any termination of this Agreement,
# Licensee will delete any and all copies of the Software. You agree that all
# provisions which operate to protect the proprietary rights of Licensor shall
# remain in force should breach occur and that the obligation of
# confidentiality described in this Agreement is binding in perpetuity and, as
# such, survives the term of the Agreement.
#
# FEE: Provided Licensee abides completely by the terms and conditions of this
# Agreement, there is no fee due to Licensor for Licensee's use of the
# Software in accordance with this Agreement.
#
# DISCLAIMER OF WARRANTIES:  THE SOFTWARE IS PROVIDED "AS-IS" WITHOUT WARRANTY
# OF ANY KIND INCLUDING ANY WARRANTIES OF PERFORMANCE OR MERCHANTABILITY OR
# FITNESS FOR A PARTICULAR USE OR PURPOSE OR OF NON- INFRINGEMENT.  LICENSEE
# BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE SOFTWARE AND
# RELATED MATERIALS.
#
# SUPPORT AND MAINTENANCE: No Software support or training by the Licensor is
# provided as part of this Agreement.
#
# EXCLUSIVE REMEDY AND LIMITATION OF LIABILITY: To the maximum extent
# permitted under applicable law, Licensor shall not be liable for direct,
# indirect, special, incidental, or consequential damages or lost profits
# related to Licensee's use of and/or inability to use the Software, even if
# Licensor is advised of the possibility of such damage.
#
# EXPORT REGULATION: Licensee agrees to comply with any and all applicable
# export control laws, regulations, and/or other laws related to embargoes and
# sanction programs administered by law.
#
# SEVERABILITY: If any provision(s) of this Agreement shall be held to be
# invalid, illegal, or unenforceable by a court or other tribunal of competent
# jurisdiction, the validity, legality and enforceability of the remaining
# provisions shall not in any way be affected or impaired thereby.
#
# NO IMPLIED WAIVERS: No failure or delay by Licensor in enforcing any right
# or remedy under this Agreement shall be construed as a waiver of any future
# or other exercise of such right or remedy by Licensor.
#
# GOVERNING LAW: This Agreement shall be construed and enforced in accordance
# with the laws of Germany without reference to conflict of laws principles.
# You consent to the personal jurisdiction of the courts of this country and
# waive their rights to venue outside of Germany.
#
# ENTIRE AGREEMENT AND AMENDMENTS: This Agreement constitutes the sole and
# entire agreement between Licensee and Licensor as to the matter set forth
# herein and supersedes any previous agreements, understandings, and
# arrangements between the parties relating hereto.
#
#        THIS HEADER MAY NOT BE EXTRACTED OR MODIFIED IN ANY WAY.
"""
Implements methods that help with handling the BERT models.
Note: Usage of
        # pylint: disable=not-callable
        # pylint: disable=no-member
to remove pytorch error warnings that ocur pre 1.0.1 where accessing torch causes these issues.
"""

import logging
import os
import re
import random

import numpy as np

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig

from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

LOGGER = logging.getLogger(__name__)


def save_model(bison_args, model, prefix=None):
    """
    Saves a model.

    :param bison_args: instance of :py:class:BisonArguments
    :param model: the model to save
    :param prefix: the prefix to attach to the file name
    :return: the location of the output file
    """
    model_to_save = model.module if hasattr(model, 'module') else model
    # Only save the model it-self
    if prefix:
        output_model_file = os.path.join(bison_args.output_dir, "%s.pytorch_model.bin" % prefix)
    else:
        output_model_file = os.path.join(bison_args.output_dir, "pytorch_model.bin")
    torch.save(model_to_save.state_dict(), output_model_file)
    return output_model_file


def set_seed(seed):
    """
    Sets the seed.

    :param seed: seed to set, set -1 to draw a random number
    :return: 0 on success
    """
    if seed == -1:
        seed = random.randrange(2**32 - 1)
    LOGGER.info("Seed: %s", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return 0


def create_tensor_dataset(data_handler):
    """
    Using a data_handler, whose features have been filled via the function
    convert_examples_to_features from a subclass instance of :py:class:Masking,
    convert the features into a TensorDataset
    :param data_handler: instance or subclass instance of :py:class:Bitext
    :return: the features represented as a TensorDataset
    """
    # pylint: disable=not-callable
    # pylint: disable=no-member
    all_input_ids = torch.tensor([f.input_ids for f in data_handler.features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in data_handler.features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in data_handler.features], dtype=torch.long)
    all_gen_label_ids = torch.tensor([f.gen_label_ids for f in data_handler.features],
                                     dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    data_set = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                             all_gen_label_ids, all_example_index)
    return data_set


def get_tokenizer(bison_args):
    """
    Based on the command line arguments, gets the correct Bert tokenizer.

    :param bison_args: an instance of :py:class:BisonArguments
    :return: a Bert tokenizer
    """
    if bison_args.bert_tokenizer is None:  # then tokenizer is the same name as the bert model
        if bison_args.bert_model == 'bert-vanilla':
            raise ValueError('For the bert-vanilla model, '
                             'a seperate bert_tokenizer must be specified.')
        tokenizer = BertTokenizer.from_pretrained(bison_args.bert_model,
                                                  do_lower_case=bison_args.do_lower_case)
    else:  # then a separate tokenizer has been specified
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased",
                                                  do_lower_case=bison_args.do_lower_case)
    return tokenizer


def prepare_train_data_loader(bison_args, masker, data_handler, tokenizer):
    """
    Prepares the TensorDataset for training.

    :param bison_args: instance of :py:class:BisonArguments
    :param masker: the masker which will mask the data as appropriate, an instance of a subclass of
    :py:class:Masking
    :param data_handler: the dataset handler, an instance of :py:class:BitextHandler or a subclass
    :param tokenizer: the BERT tokenizer
    :return: train_dataloader, an instance of :py:class:TensorDataset
    """
    masker.convert_examples_to_features(
        data_handler=data_handler,
        tokenizer=tokenizer,
        max_seq_length=bison_args.max_seq_length,
        max_part_a=bison_args.max_part_a,
        is_training=True)

    train_data = create_tensor_dataset(data_handler)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                  batch_size=bison_args.train_batch_size)
    return train_dataloader


def load_model(bison_args, device, data_handler, output_model_file=None):
    """
    Load a model.

    :param bison_args: instance of :py:class:BisonArguments
    :param device: the device to move the model to
    :param data_handler: the dataset handler, an instance of :py:class:BitextHandler or a subclass
    :param output_model_file: the location of the model to load
    :return: the loaded model
    """

    model_state_dict = None
    if output_model_file is not None:
        model_state_dict = torch.load(output_model_file)

    if bison_args.bert_model == 'bert-vanilla':
        # randomly initialises BERT weights instead of using a pre-trained model
        model = BertForMaskedLM(BertConfig.from_default_settings())
    else:
        model = BertForMaskedLM.from_pretrained(bison_args.bert_model, state_dict=model_state_dict)
    model.to(device)
    return model


def argument_sanity_check(bison_args):
    """
    Performs a couple of additional sanity check on the provided arguments.

    :param bison_args: instance of :py:class:BisonArguments
    :return: 0 on success (else an error is raise)
    """
    if bison_args.do_train:
        if not bison_args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if bison_args.do_predict:
        if not bison_args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(os.path.join(bison_args.output_dir, "pytorch_model.bin")) \
            and bison_args.do_train:
        if not bison_args.load_prev_model:
            raise ValueError("Output directory already contains a saved model (pytorch_model.bin).")
    os.makedirs(bison_args.output_dir, exist_ok=True)
    return 0


def prepare_optimizer(bison_args, model, num_train_steps):
    """
    Prepares the optimizer for training.
    :param bison_args: instance of :py:class:BisonArguments
    :param model: the model for which the optimizer will be created
    :param num_train_steps: the total number of training steps that will be performed
            (need for learning rate schedules that depend on this)
    :return: the optimizer and the number of total steps
    """
    param_optimizer = list(model.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]

    t_total = num_train_steps
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=bison_args.learning_rate,
                         warmup=0.1,
                         t_total=t_total)

    return optimizer, t_total
