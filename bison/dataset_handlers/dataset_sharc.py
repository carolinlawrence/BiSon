# coding=utf-8
#        BiSon
#
#   File:     dataset_sharc.py
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
Implements the ShARC dataset (https://sharc-data.github.io).
"""

import logging
import os
import json
from collections import Counter

from bison.util import write_list_to_file, write_json_to_file
from bison.evals.evaluator_sharc import evaluate as evaluate_sharc
from .dataset_bitext import BitextHandler, GenExample

LOGGER = logging.getLogger(__name__)


class SharcHandler(BitextHandler):
    """
    Handles the ShARC dataset (https://sharc-data.github.io).
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self):
        super().__init__()
        self.examples = []
        self.features = []
        self.write_predictions = write_json_to_file
        self.write_eval = write_json_to_file

    @staticmethod
    def extract_answer(in_file, out_file):
        """
        Given a Sharc json file, extract just the answers
        :param in_file: Sharc json file
        :param out_file: file with just answers
        :return: 0 on success
        """
        with open(in_file, 'r') as file:
            all_elements = json.load(file)
        new_list = []

        for ele in all_elements:
            new_list.append(ele["answer"].replace("\n", " "))

        write_list_to_file(new_list, out_file)
        return 0

    def read_examples(self, input_file, is_training=False):
        """
        Reads a Sharc dataset, each instance in self.examples holds a :py:class:SharcExample object
        :param input_file: the file containing Sharc data (json)
        :param is_training: True for training, then we read in gold labels, else we do not.
        :return: 0 on success
        """
        self.examples = []  # reset previous lot
        LOGGER.info("Part a: question + ruletext + scenario + history")
        LOGGER.info("Part b: answer")
        with open(input_file, "r", encoding='utf-8') as reader:
            all_data = json.load(reader)

        example_counter = 0
        for instance in all_data:

            answer_text = ""
            if is_training is True:
                answer_text = instance["answer"]

            history = []
            for previous_utt in instance["history"]:
                if "follow_up_question" in previous_utt and "follow_up_answer" in previous_utt:
                    history.append([previous_utt["follow_up_question"],
                                    previous_utt["follow_up_answer"]])
                else:
                    logging.info("Warning: key error in: %s", previous_utt)
            evidence = []
            for previous_utt in instance["evidence"]:
                if "follow_up_question" in previous_utt and "follow_up_answer" in previous_utt:
                    evidence.append([previous_utt["follow_up_question"],
                                     previous_utt["follow_up_answer"]])
                else:
                    logging.info("Warning: key error in: %s", previous_utt)

            example = SharcHandler.SharcExample(
                example_index=example_counter,
                utt_id=instance["utterance_id"],
                tree_id=instance["tree_id"],
                source_url=instance["source_url"],
                ruletext=instance["snippet"],
                question_text=instance["question"],
                scenario=instance["scenario"],
                answer_text=answer_text,
                history=history,
                evidence=evidence)
            self.examples.append(example)
            example_counter += 1
        return 0

    def arrange_generated_output(self, current_example, generated_text):
        """
        Reproduces the json output structure that the sharc evaluator expects
        :param current_example: The current example (so that necessary components for writing
        can be accessed, e.g. unique id)
        :param generated_text: the text generated by the model
        :return: a json object
        """
        return {"utterance_id": current_example.utt_id, "answer": generated_text}

    def arrange_token_classify_output(self, current_example, classification_tokens,
                                      input_ids, tokenizer):
        """
        Simply returns all classification elements, other data sets can arrange the output
        as needed here.
        :param current_example: The current example
        :param classification_tokens: the classification labels for all tokens
        :return: classification_tokens in a dictionary so we get json
        """
        return classification_tokens

    def evaluate(self, output_prediction_file, valid_gold, mode='generation'):
        """
        Given the location of the prediction and gold output file,
        calls a dataset specific evaluation script.
        Here calls the Sharc evaluation script.
        :param output_prediction_file: the file location of the predictions
        :param valid_gold: the file location of the gold outputs
        :param mode: if generation, calls combined evaluation
        :return: a dictionary, for classify it has the keys micro_accuracy and macro_accuracy
                for combined additionally bleu_* where * in {1,2,3,4}
        """
        sharc_mode = 'combined'
        self.extract_answer(output_prediction_file, output_prediction_file+".txt")
        self.extract_answer(valid_gold, output_prediction_file+".gold")
        bitext_results = super(SharcHandler, self).evaluate(
            output_prediction_file+".txt", output_prediction_file+".gold", mode=mode)
        os.remove(output_prediction_file+".gold")
        try:
            results = evaluate_sharc(valid_gold, output_prediction_file, sharc_mode)
        except ZeroDivisionError:  # if results are really bad, the evaluate script throws an error
            results = {'micro_accuracy': 0.0, 'macro_accuracy': 0.0, 'bleu_1': 0.0, 'bleu_2': 0.0,
                       'bleu_3': 0.0, 'bleu_4': 0.0}
        for key in results:
            bitext_results[mode+"_"+key] = results[key]
        return bitext_results

    def select_deciding_score(self, results):
        """
        Select the deciding score for saving models.
        :param results: the dictionary returned by the evaluate call
        :return: bleu_4
        """
        deciding_score = results['generation_bleu_4']
        return deciding_score

    class SharcExample(GenExample):
        """A single training/test example from the Sharc corpus.
        """
        # pylint: disable=too-many-arguments
        def __init__(self,
                     example_index,
                     utt_id,
                     tree_id,
                     source_url,
                     ruletext,
                     question_text,
                     scenario,
                     answer_text,
                     history,  # a list of lists where each instance is another list with two items
                     # [follow_up_question, follow_up_answer]
                     evidence):  # evidence has same structure as history
            super().__init__()
            self.example_index = example_index
            self.utt_id = utt_id
            self.tree_id = tree_id
            self.source_url = source_url
            self.ruletext = ruletext
            self.question_text = question_text
            self.scenario = scenario
            self.answer_text = answer_text
            self.history = history
            self.evidence = evidence

            self.part_b = self.answer_text
            self.part_a = ""
            self.part_a += self.question_text + "\n"
            self.part_a += self.ruletext + "\n" + self.scenario
            if history:
                self.part_a += "\n"
            for element in history:
                self.part_a += element[0] + "\n" + element[1] + "\n"
