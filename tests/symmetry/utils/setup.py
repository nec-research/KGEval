# coding: utf-8
#             KGEval
#
#   File:     setup.py
#   Authors:  Wiem Ben Rim wiem.benrim@nlp.c.titech.ac.jp
#             Carolin Lawrence carolin.lawrence@neclab.eu
#             Kiril Gashteovski kiril.gashteovski@neclab.eu
#             Mathias Niepert mathias.niepert@neclab.eu
#             Naoaki Okazaki okazaki@c.titech.ac.jp
#
# NEC Laboratories Europe GmbH, Copyright (c) 2021, All rights reserved.
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
#
#

"""
Loads model and runs symmetry behavioral tests
"""


from pykeen.triples import TriplesFactory 
import pandas as pd
import collections
import torch
import pathlib
import sys
from symmetry.utils import logger as l

"""
Knowledge Graph dataset pre-processing functions
"""
def get_filters(examples):
  """
  Returns rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
  Params
  ----------
  examples: Numpy array of size n_examples x 3 containing KG triples
  """
  rhs_filters = collections.defaultdict(set)
  for lhs, rel, rhs in examples:
      rhs_filters[(lhs, rel)].add(rhs)
  rhs_final = {}
  for k, v in rhs_filters.items():
      rhs_final[k] = sorted(list(v))
  return rhs_final

def process_data(data, dataset_name, tests_path):
  """
  data is a dictionary with keys ["train", "test", "valid"]
  """
  data_dfs = {}
  for split in ["train", "test", "valid"]:
    data_dfs[split] = pd.read_csv(data[dataset_name][split], sep="\t", header=None)
  all_examples = []
  for split in ["train", "test", "valid"]:
    all_examples.extend(data_dfs[split].values)
  test_splits = ["test2.csv", "test3.csv"] #they do not necessarily occur in the test set
  for split in test_splits:
    path = tests_path /split
    all_examples.extend(pd.read_csv(path, sep="\t", header=None).values)
  filters = get_filters(all_examples)
  return filters


"""
Pykeen model loading
"""
def get_triples(data, test_location, dataset_name, model_name):
  """
  Returns training and testing triples, as well as the model from the model location
  Params
  -----------
  test_location: a string of the location of a test set
  model_location: a string of the location of the model you want to test
  """
  training = TriplesFactory.from_path(data[dataset_name]["train"].absolute().as_posix())
  testing = TriplesFactory.from_path(
      test_location,
      entity_to_id=training.entity_to_id,
      relation_to_id=training.relation_to_id,
  )
  model_path = (pathlib.Path.cwd() / 'models' / dataset_name / model_name).joinpath("trained_model.pkl")
  try:
    model = torch.load(model_path)
  except FileNotFoundError:
    print(f"FileNotFoundError: Cannot find model location, did you add the model {model_name} under models/{dataset_name}? ")
    raise
  triples = testing.triples
  return training, triples, model

def print_progress_bar(index, total, label):
    n_bar = 50  
    progress = index / total
    sys.stdout.write('\r')
    sys.stdout.write(f"[{'=' * int(n_bar * progress):{n_bar}s}] {int(100 * progress)}%  {label}")
    sys.stdout.flush()

def get_prediction_results(logger, data, model_name, tests_path, test, dataset_name):
  """
  Runs predictions on the test set with the given model
  Returns an array of results of the following format for each test set sample [head, relation, top prediction, rank, reciprocal, ground truth, softmax score]
  Prams
  -----------
  test_location: a string of the location of a test set
  model_location: a string of the location of the model you want to test
  """
  logger.info(f"Starting {test} with {model_name} model") 
  ranks = []
  dataset_filters = process_data(data, dataset_name, tests_path)
  current_test_path = tests_path.joinpath(test+".csv").absolute().as_posix()
  training, triples, model = get_triples(data, current_test_path, dataset_name, model_name)
  for t in range(len(triples)):
    if triples[t][0] in training.entity_to_id.keys() and triples[t][1] in training.relation_to_id.keys():
      head, relation, expected_tail = triples[t][0], triples[t][1], triples[t][2]
      _predictions = model.get_tail_prediction_df(head, relation,triples_factory=training)['tail_label'].values

      filter_out = list(filter(lambda prediction: prediction != expected_tail,  dataset_filters.get((head, relation), [])))
      filtered = list(filter(lambda prediction: prediction not in filter_out, _predictions))

      for i in range(len(filtered)): #this loop looks for the rank of the expected tail in the array of predictions
        if filtered[i] == expected_tail:
            ranks.append(i+1)
            break
    print_progress_bar(t, len(triples), "Triples tested")
  return ranks


"""
Processing prediction results from pykeen
"""
def report_results(logger, data, dataset_name, model_name, test, symmetry=True):
  """
  Takes test results and returns the MRR and hits@1, hits@5, hits@10 and array of entropy
  -----------
  Params
  model_location: a string of the location of the model you want to test
  test_location: a string of the location of a test set
  test_type: string describing the type of test (e.g test1, memorization, etc..)
  """
  path = pathlib.Path.cwd()
  tests_path = path / 'behavioral_tests' / dataset_name / 'symmetry' 
  ranks = get_prediction_results(logger, data, model_name, tests_path, test, dataset_name)
  mean_rank, mean_reciprocal_rank, hits_at = get_metrics(ranks, symmetry)
  list1=[test, model_name, mean_rank, mean_reciprocal_rank, hits_at[0].item(),hits_at[1].item(),hits_at[2].item()]
  p = path / 'results' / dataset_name / 'symmetry'
  p.mkdir(parents=True, exist_ok=True)
  results_path = path / 'results' / dataset_name / 'symmetry' / (test+'.csv')
  with results_path.open("w", encoding ="utf-8") as f:
    f.write(f"Results of {model_name} model on {dataset_name} dataset\n: on {test}, MR: {mean_rank}, MRR: {mean_reciprocal_rank}, Hits@1: {hits_at[0]}, Hits@5: {hits_at[1]}, Hits@10: {hits_at[2]}")
  print("\n")
  logger.info(f"On {test}, MR: {mean_rank}, MRR: {mean_reciprocal_rank}, Hits@1: {hits_at[0]}, Hits@5: {hits_at[1]}, Hits@10: {hits_at[2]}")

def get_metrics(ranks, symmetric=True):
  """
  Params
  ----------
  ranks: list of ranks of the correct tail for each testing triple
  symmetric: true if test1-3, false if test4
  """
  ranks = torch.tensor(ranks).to(dtype=float)
  mean_rank = torch.mean(ranks).item()
  mean_reciprocal_rank = torch.mean(1. / ranks).item()
  if symmetric:
    hits_at = torch.FloatTensor((list(map(
        lambda x: torch.mean((ranks <= x).float()).item(),
        (1, 5, 10)
    ))))
  else:
    hits_at = torch.FloatTensor((list(map(
        lambda x: torch.mean((ranks > x).float()).item(),
        (1, 5, 10)
    ))))
  return mean_rank, mean_reciprocal_rank, hits_at


def test(data, dataset_name, model_name):
  """
  Run symmetry behavioral tests 
  """
  logger = l.get_logger(__name__)
  for test in ["test1", "test2", "test3"]:
    report_results(logger, data, dataset_name, model_name,test, dataset_name)
  report_results(logger, data, dataset_name, model_name, "test4",False)
