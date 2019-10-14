# coding=utf-8
#        BiSon
#
#   File:     train.py
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
Handles training of BiSon models.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from tqdm import tqdm, trange

from .predict import predict
from .model_helper import save_model, prepare_train_data_loader, prepare_optimizer

LOGGER = logging.getLogger(__name__)


def get_loss(model, batch):
    """
    Given a batch, gets the loss for the chosen BERT model.

    :param bison_args: an instance of :py:class:BisonArguments
    :param model: a BERT model
    :param batch: the current batch
    :return: the loss of the current batch for the chosen model.
    """
    input_ids, input_mask, segment_ids, gen_label_ids, _ = batch

    # Get loss
    loss = model(input_ids, segment_ids, input_mask, gen_label_ids)
    return loss


def train(bison_args, data_handler, data_handler_predict, model, masker, tokenizer, device):
    """
    Runs training for a model.

    :param bison_args: Instance of :py:class:BisonArguments
    :param data_handler: instance or subclass of :py:class:Bitext, for training
    :param data_handler_predict: instance or subclass of :py:class:Bitext, for validation
    :param model: the model that will be trained
    :param masker: subclass instance of :py:class:Masking
    :param tokenizer: instance of BertTokenzier
    :param device: the device to run the computation on
    :return: 0 on success
    """
    train_examples = data_handler.examples
    num_train_steps = \
        int(len(train_examples) / bison_args.train_batch_size /
            bison_args.gradient_accumulation_steps * bison_args.num_train_epochs)

    optimizer, t_total = prepare_optimizer(bison_args, model, num_train_steps)

    train_dataloader = prepare_train_data_loader(bison_args, masker, data_handler, tokenizer)

    best_valid_score = 0.0  # if validation is run during training, keep track of best

    model.train()
    n_params = sum([p.nelement() for p in model.parameters()])
    LOGGER.info("Number of parameters: %d", n_params)

    for epoch in trange(int(bison_args.num_train_epochs), desc="Epoch"):
        LOGGER.info("Starting Epoch %s:", epoch)

        # some masking changes at every epoch, thus reload if necessary
        if bison_args.masking_strategy is not None and epoch != 0:  # already done for first epoch
            LOGGER.info("Recreating masks")
            train_dataloader = prepare_train_data_loader(bison_args,
                                                         masker,
                                                         data_handler,
                                                         tokenizer)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(instance.to(device) for instance in batch)  # move batch to gpu

            loss = get_loss(model, batch)  # access via loss.item()

            if bison_args.gradient_accumulation_steps > 1:
                loss = loss / bison_args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % bison_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        # Validate on the dev set if desired.
        if bison_args.valid_every_epoch:
            best_valid_score = validate(best_valid_score, bison_args, data_handler_predict, masker,
                                        tokenizer, model, device, epoch)

    # save last model if we didn't pick the best during training
    if not bison_args.valid_every_epoch:
        LOGGER.info("Saving final model")
        save_model(bison_args, model)

    return best_valid_score


def validate(best_valid_score, bison_args, data_handler_predict, masker, tokenizer, model, device,
             epoch):
    """
    After an epoch of training, validate on the validation set.

    :param best_valid_score: the currently best validation score
    :param bison_args: an instance of :py:class:BisonArguments
    :param data_handler_predict: instance or subclass instance of :py:class:Bitext,
     on which to run prediction
    :param masker: an instance of a subclass of :py:class:Masking
    :param tokenizer: the BERT tokenizer
    :param model: the BERT model
    :param device: where to run computations
    :param epoch: the current epoch
    :return: the new best validation score
    """
    model.eval()
    if best_valid_score == 0.0:  # then first epoch, save model
        save_model(bison_args, model)
    deciding_score = predict(bison_args, data_handler_predict, masker, tokenizer, model,
                             device, epoch)
    if best_valid_score < deciding_score:
        LOGGER.info("Epoch %s: Saving new best model: %s vs. previous %s",
                    epoch, deciding_score, best_valid_score)
        save_model(bison_args, model)
        best_valid_score = deciding_score
    model.train()
    return best_valid_score
