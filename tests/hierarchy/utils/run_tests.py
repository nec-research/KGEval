
# coding: utf-8
#             KGEval
#
#   File:     run_tests.py
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
Runs hierarchy tests 
"""
from pykeen.triples import TriplesFactory 
from csv import writer
import numpy as np
import pandas as pd
import collections
import pathlib
import torch
import json
import os
from symmetry.utils import logger as l

MAX_DEPTH = 4

"""
Knowledge Graph dataset pre-processing functions.
"""

def to_np_array(dataset_file):
    """Map raw dataset file to numpy array with unique ids.
    Args:
      dataset_file: Path to file containing raw triples in a split
      ent2idx: Dictionary mapping raw entities to unique ids
      rel2idx: Dictionary mapping raw relations to unique ids
    Returns:
      Numpy array of size n_examples x 3 mapping the raw dataset file to ids
    """
    examples = []
    with open(dataset_file, "r") as lines:
        for line in lines:
            lhs, rel, rhs = line.strip().split("\t")
            try:
                examples.append([lhs, rel, rhs])
            except ValueError:
                continue
    return np.array(examples)


def get_filters(examples, n_relations="237"):
    """Create filtering lists for evaluation.
    Args:
      examples: Numpy array of size n_examples x 3 containing KG triples
      n_relations: Int indicating the total number of relations in the KG
    Returns:
      lhs_final: Dictionary mapping queries (entity, relation) to filtered entities for left-hand-side prediction
      rhs_final: Dictionary mapping queries (entity, relation) to filtered entities for right-hand-side prediction
    """
    lhs_filters = collections.defaultdict(set)
    rhs_filters = collections.defaultdict(set)
    for lhs, rel, rhs in examples:
        rhs_filters[(lhs, rel)].add(rhs)
        lhs_filters[(rhs, rel + n_relations)].add(lhs)
    lhs_final = {}
    rhs_final = {}
    for k, v in lhs_filters.items():
        lhs_final[k] = sorted(list(v))
    for k, v in rhs_filters.items():
        rhs_final[k] = sorted(list(v))
    return lhs_final, rhs_final


def process_dataset(path, test_path):
    """Map entities and relations to ids and saves corresponding pickle arrays.
    Args:
      path: Path to dataset directory
    Returns:
      examples: Dictionary mapping splits to with Numpy array containing corresponding KG triples.
      filters: Dictionary containing filters for lhs and rhs predictions.
    """
    
    examples = {}
    splits = ["train.txt", "valid.txt", test_path]
    for split in splits:
        if split == "train.txt" or split == "valid.txt":
            dataset_file = os.path.join(path, split)
            examples[split] = to_np_array(dataset_file)
        else:
            dataset_file = split
            examples[split] = to_np_array(dataset_file)
    all_examples = np.concatenate([examples[split] for split in splits], axis=0)
    lhs_skip, rhs_skip = get_filters(all_examples)
    filters = {"rhs": rhs_skip}
    return examples, filters

def get_triples(train_path, test_path, model_path):
    """
    Returns training and testing triples, as well as the model from the model location
    Params
    -----------
    test_location: a string of the location of a test set
    model_location: a string of the location of the model you want to test
    """
    training = TriplesFactory(path=str(train_path))
    testing = TriplesFactory(
        path= test_path,
        entity_to_id=training.entity_to_id,
        relation_to_id=training.relation_to_id,
    )
    model = torch.load(model_path)
    triples = testing.triples
    return training, triples, model

def get_prediction_results(test_level, test_location, model_path):
    """
    Runs predictions on the test set with the given model
    Returns an array of results of the following format for each test set sample [head, relation, top prediction, rank, reciprocal, ground truth, softmax score]
    Prams
    -----------
    test_location: a string of the location of a test set
    model_location: a string of the location of the model you want to test
    """
    
    resource_path = pathlib.Path.cwd()/ 'tests' / 'hierarchy' / 'resources' 
    # The below resource loads the types expected for all relations from the training set
    with open(resource_path / "leveled_ranges.txt", 'r') as g: 
        leveled_ranges = json.load(g)
    # The below resource informs the type of an entity at different levels 
    # (Tom Cruise at Level 4 is Actor, at Level 1 is Agent)
    level_up_path = pathlib.Path.cwd()/ 'tests' / 'hierarchy' / 'resources' / 'level_up_map'
    entity_to_type_at_level =  {str(k): v for k, v in zip(range(MAX_DEPTH+1), [{},{},{},{},{}])}
    
    #Load the map that links entities to their types at given levels
    for level in range(MAX_DEPTH+1):
        with open(os.path.join(level_up_path, 'level_' + str(level)+'.txt'), 'r') as g: 
            entity_to_type_at_level[str(level)] = json.load(g)

    data_path = pathlib.Path.cwd() / "data" /"FB15K237" 
    train_path = data_path / "train.txt"

    _, dataset_filters = process_dataset(data_path, test_location)
    training, triples, model = get_triples(train_path, test_location, model_path)
    ranks = []
    k_level_ranks =  {str(k): v for k, v in zip(range(MAX_DEPTH+1), [[],[],[],[],[]])}
    
    for t in range(len(triples)): 
        if triples[t][0] in training.entity_to_id.keys() and triples[t][1] in training.relation_to_id.keys():
            head, relation, expected_tail = triples[t][0], triples[t][1], triples[t][2]
            _predictions = pd.DataFrame(model.predict_tails(str(head), str(relation)).to_numpy())[1].values
            filter_out = list(filter(lambda prediction: prediction != expected_tail, dataset_filters["rhs"].get((head, relation), [])))
            filtered = list(filter(lambda prediction: prediction not in filter_out, _predictions))

            for i in range(len(filtered)): 
                if filtered[i] == expected_tail:
                    ranks.append(i+1)
                    break          
        
            current_level = test_level 
            while int(current_level)>0: 
                found = False
                same_type_count = 0
                for i in range(len(filtered)):                   
                    same_level_entities = entity_to_type_at_level[current_level]
                    if filtered[i] in same_level_entities.keys() and expected_tail in same_level_entities.keys():
                        pred_type = same_level_entities[filtered[i]] 
                        for k in range(len(leveled_ranges[relation][current_level])):                              
                            expected_type = leveled_ranges[relation][current_level][k]
                            if expected_type == pred_type:
                                same_type_count =  same_type_count + 1
                                if filtered[i] == expected_tail:
                                    k_level_ranks[current_level].append(same_type_count)
                                    found = True
                                    break        
             
                if found == False: 
                    for j in range(len(filtered)): 
                        if filtered[j] == expected_tail:
                            found = True
                            k_level_ranks[current_level].append(j+1)
                            break
                            
                current_level = str(int(current_level)-1)
                

    return ranks, k_level_ranks


def get_metrics(ranks):
    ranks = torch.tensor(ranks).to(dtype=float)
    mean_rank = torch.mean(ranks).item()
    mean_reciprocal_rank = torch.mean(1. / ranks).item()
    
    hits_at = torch.FloatTensor((list(map(
        lambda x: torch.mean((ranks <= x).float()).item(),
        (1, 5, 10)
    ))))
    return mean_rank, mean_reciprocal_rank, hits_at

def report_results_top_k(logger, model_location, test_data, test_level, print_to_console=True):
    """
    Takes test results and returns the MRR and hits@1, hits@5, hits@10 and array of entropy
    -----------
    Params
    model_location: a string of the location of the model you want to test
    test_location: a string of the location of a test set
    test_type: string describing the type of test (e.g test1, memorization, etc..)
    """
    ranks, k_level_ranks = get_prediction_results(test_level, test_data, model_location)
    mean_rank, mean_reciprocal_rank, hits_at = get_metrics(ranks)
    list1=[test_level, "gold_tail", model_location, mean_reciprocal_rank, hits_at[0].item(),hits_at[1].item(),hits_at[2].item()]
    results_path =  pathlib.Path.cwd()/ 'results' / 'FB15K237' / 'hierarchy'
    results_path.mkdir(parents=True, exist_ok=True)
    file_path = os.path.join(results_path, test_level+'.csv')
    results_dict = {}
    with open(file_path, 'a') as f:
        writer_object = writer(f)
        writer_object.writerow(["TEST LEVEL", "HIERARCHY TEST", "MODEL", "MRR", "HITS@1", "HITS@5", "HITS@10"])
        writer_object.writerow(list1)
        logger.info(f"""TEST LEVEL: {test_level}\nHIERARCHY TEST1: GOLD TAIL\nMODEL: {model_location}\nMRR {list1[3]}\nHits@1: {list1[4]}\nHits@5: {list1[5]}\nHits@10: {list1[6]}\n""")
        for level in range(1, int(test_level)+1):
            _mean_rank, _mean_reciprocal_rank, _hits_at = get_metrics(k_level_ranks[str(level)])
            list2=[test_level, level, model_location, _mean_reciprocal_rank,_hits_at[0].item(),_hits_at[1].item(),_hits_at[2].item()]
            if level == int(test_level): 
                logger.info(f"""TEST LEVEL: {test_level}\nHIERARCHY TEST2: TYPE CONSTRAINTS AT LEVEL {level}\nMODEL: {model_location}\nMRR {list2[3]}\nHits@1: {list2[4]}\nHits@5: {list2[5]}\nHits@10: {list1[6]}\n""")                                                
            results_dict[level] = list2
            writer_object.writerow(list2)
    f.close()
    return list1, results_dict


def test(dataset_name, model_name):
    logger = l.get_logger(__name__)
    hierarchy_tests_path = pathlib.Path.cwd()/ 'behavioral_tests' / dataset_name / 'hierarchy' 
    if dataset_name == "FB15K237":
        for test in range(MAX_DEPTH+1):
            current_test_path = os.path.join(hierarchy_tests_path, str(test)+'.txt')
            model_path = pathlib.Path.cwd()/ 'models' / 'FB15K237' / model_name / 'trained_model.pkl'
            results_dict = report_results_top_k(logger, model_path, current_test_path, str(test))
    else: 
        logger.info("THIS TEST IS ONLY AVAILABLE FOR THE FB15K237 DATASET")
