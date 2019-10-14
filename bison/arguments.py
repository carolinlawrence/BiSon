# coding=utf-8
#        BiSon
#
#   File:     arguments.py
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
Handles the possible command line arguments for both BiSon and GPT2.
"""

import argparse


class GeneralArguments():
    """
    Settings relevant for every training and prediction scenario
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self._add_arguments()
        bison_args = vars(self.parser.parse_args())
        for key in bison_args:
            setattr(self, key, bison_args[key])

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def _add_arguments(self):
        # Required parameters
        self.parser.add_argument("--output_dir", default=None, type=str, required=True,
                                 help="The output directory where all relevant files will"
                                      "be written to.")

        self.parser.add_argument('--valid_every_epoch',
                                 action='store_true',
                                 help="Whether to validate on the validation set after every "
                                      "epoch, save best model according to the evaluation metric "
                                      "indicated by each specific dataset class.")
        self.parser.add_argument("--load_prev_model", default=None, type=str,
                                 help="Provide a file location if a previous model should be "
                                      "loaded. (Note that Adam Optimizer paramters are lost.")

        ## Other parameters
        self.parser.add_argument("--data_set", type=str, required=True,
                                 choices=['sharc', 'daily_dialog', 'bitext'],
                                 help="Which dataset to expect.")
        self.parser.add_argument("--train_file", default=None, type=str,
                                 help="Input file for training")
        self.parser.add_argument("--predict_file", default=None, type=str,
                                 help="Input file for prediction.")
        self.parser.add_argument("--valid_gold", default=None, type=str,
                                 help="Location of gold file for evaluating predictions.")
        self.parser.add_argument("--max_seq_length", default=384, type=int,
                                 help="The maximum total sequence length (Part A + B) after "
                                      "tokenization. "
                                      "Note: For daily_dialog we truncate the beginning.")
        self.parser.add_argument("--max_part_a", default=64, type=int,
                                 help="The maximum number of tokens for Part A. Sequences longer "
                                      "than this will be truncated to this length.")
        self.parser.add_argument("--do_train", action='store_true', help="Should be true to run taining.")
        self.parser.add_argument("--do_predict", action='store_true',
                                 help="Should be true to run predictition.")
        self.parser.add_argument("--train_batch_size", default=16, type=int,
                                 help="Batch size to use for training. "
                                      "Actual batch size will be divided by "
                                      "gradient_accumulation_steps and clipped to closest int.")
        self.parser.add_argument("--predict_batch_size", default=16, type=int,
                                 help="Batch size to use for predictions.")
        self.parser.add_argument("--learning_rate", default=1e-5, type=float,
                                 help="The learning rate for Adam.")
        self.parser.add_argument("--num_train_epochs", default=3.0, type=float,
                                 help="How many training epochs to run.")
        self.parser.add_argument('--seed',
                                 type=int,
                                 default=42,
                                 help="Random seed for initialization, "
                                      "set to -1 to draw a random number.")
        self.parser.add_argument('--gradient_accumulation_steps',
                                 type=int,
                                 default=1,
                                 help="Number of updates steps to accumulate before performing "
                                      "a backward/update pass.")
        self.parser.add_argument("--do_lower_case",
                                 action='store_true',
                                 help="Whether to lower case the input text. "
                                      "Should be True for uncased models, False for cased models.")


class BisonArguments(GeneralArguments):
    """
    Arguments relevant for generating with BERT."""

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.__class__.__name__

    def _add_arguments(self):
        super(BisonArguments, self)._add_arguments()
        # Required parameters
        self.parser.add_argument("--bert_model", default=None, type=str, required=True,
                                 choices=['bert-large-uncased', 'bert-base-uncased',
                                          'bert-base-cased', 'bert-large-cased',
                                          'bert-base-multilingual-uncased',
                                          'bert-base-multilingual-cased',
                                          'bert-base-chinese', 'bert-vanilla'],
                                 help="Bert pre-trained model to use.")
        self.parser.add_argument("--bert_tokenizer", default=None, const=None, nargs='?',
                                 type=str,
                                 choices=['bert-large-uncased', 'bert-base-cased',
                                          'bert-large-cased',
                                          'bert-base-multilingual-uncased',
                                          'bert-base-multilingual-cased',
                                          'bert-base-chinese', 'bert-vanilla'],
                                 help="If the tokenizer should differ from the model, "
                                      "e.g. when initializing weights randomly but still want to "
                                      "use the vocabulary of a pre-trained BERT model.")

        #pertaining masking
        self.parser.add_argument("--masking", default=None, type=str, required=True,
                                 choices=['gen'],
                                 help="Selection of: 'gen' for generating sequences.")
        self.parser.add_argument("--max_gen_length", default=50, type=int,
                                 help="Maximum length for output generation sequence (Part B).")
        # for GenerationMasking and RandomAllPartsMasking
        self.parser.add_argument("--masking_strategy", default=None, type=str, const=None,
                                 nargs='?',
                                 choices=['bernoulli', 'gaussian'],
                                 help="Which masking strategy to us, options are: "
                                      "bernoulli, gaussian")
        self.parser.add_argument("--distribution_mean", default=1.0, type=float,
                                 help="The mean (for Bernoulli and Gaussian sampling).")
        self.parser.add_argument("--distribution_stdev", default=0.0, type=float,
                                 help="The standard deviation (for Gaussian sampling).")

        #pertaining prediction
        self.parser.add_argument("--predict", type=str, default='one_step_greedy',
                                 const='one_step_greedy',
                                 nargs='?',
                                 choices=['one_step_greedy', 'left2right', 'max_probability',
                                          'min_entropy', 'right2left', 'no_look_ahead'],
                                 help="How perdiction should be run.")
