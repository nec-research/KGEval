# KGEval
A framework for evaluating Knowledge Graph Embedding Models in a fine-grained manner.

The framework and experimental results are described in [Ben Rim et al. 2021](https://openreview.net/pdf?id=3_2B2MliB8V) (Outstanding Paper Award, AKBC 2021).

# Instructions 

## Create a virtual environment 

	virtualenv -p python3.6 eval_env
	source eval_env/bin/activate
	pip install -r requirements.txt

## Download data 
In the main folder, run:
	
	source data/download.sh

## Download model
If you want to test the framework immediately, you can download pre-trained [Pykeen](https://github.com/pykeen/pykeen) models by running:

    source download_model.sh

## Generate behavioral tests

### Symmetry Tests 
Can choose `--dataset` FB15K237, WN18RR, YAGO310

	python tests/run.py --dataset FB15K237 --mode generate --capability symmetry

This should result into the following output, and the files for each test set will be added under behavioral\_tests\dataset\symmetry: 

	2021-10-03 23:37:35,060 - [INFO] - Preparing test sets for the dataset FB15K237
	2021-10-03 23:37:37,621 - [INFO] - ########################## <----TRAIN---> ############################
	2021-10-03 23:37:37,621 - [INFO] - 0 repetitions removed
	2021-10-03 23:37:37,621 - [INFO] - 272115 triples remaining in train set
	2021-10-03 23:37:37,621 - [INFO] - 6778 symmetric triples found in train set
	2021-10-03 23:37:37,786 - [INFO] - ########################## <----TEST---> ############################
	2021-10-03 23:37:37,786 - [INFO] - 0 repetitions removed
	2021-10-03 23:37:37,786 - [INFO] - 20466 triples remaining in test set
	2021-10-03 23:37:37,786 - [INFO] - 113 symmetric triples found in test set
	2021-10-03 23:37:37,806 - [INFO] - ########################## <----VALID---> ############################
	2021-10-03 23:37:37,806 - [INFO] - 0 repetitions removed
	2021-10-03 23:37:37,806 - [INFO] - 17535 triples remaining in valid set
	2021-10-03 23:37:37,806 - [INFO] - 113 symmetric triples found in valid set
	2021-10-03 23:37:39,106 - [INFO] - #################### <---TEST SET 1: MEMORIZATION ---> ##########################
	2021-10-03 23:37:39,106 - [INFO] - There are 5470 entries in the memorization set (occur in both directions)
	2021-10-03 23:37:39,106 - [INFO] - #################### <---TEST SET 2: ONE DIRECTION SEEN ---> ##########################
	2021-10-03 23:37:39,106 - [INFO] - There are 1308 entries not shown in both directions (to be reversed for testing)
	2021-10-03 23:37:39,836 - [INFO] - #################### <--- SYMMETRIC RELATIONS ---> ##########################
	2021-10-03 23:37:39,836 - [INFO] - TRAIN SET contains 6778 symmetric entries
	2021-10-03 23:37:39,836 - [INFO] - TEST SET contains  113 symmetric entries with 113 not in training
	2021-10-03 23:37:39,836 - [INFO] - VALID SET contains 113 symmetric entries with 113 not in training
	2021-10-03 23:37:39,839 - [INFO] - #################### <---TEST SET 3: UNSEEN INSTANCES ---> ##########################
	2021-10-03 23:37:39,840 - [INFO] - There are 226 entries that are not seen in any direction in training
	2021-10-03 23:37:40,267 - [INFO] - #################### <---TEST SET 4: ASYMMETRY ---> ##########################
	2021-10-03 23:37:40,267 - [INFO] - There are 3000 asymmetric entries in test set added to test 4

### Hierarchy Tests 
**Only available for FB15K237 dataset**

	python tests/run.py --dataset FB15K237 --mode generate --capability hierarchy

The output should be and will be available under behavioral\_tests/dataset/hierarchy/, the naming of the files corresponds to triples where the tail belongs to a specified level. For example, 1.txt contains triples where the tail has a type of level 1 in the entity type hierarchy :

	2021-10-04 01:38:13,517 - [INFO] - Results of Hierarchy Behavioral Tests for FB15K237
	2021-10-04 01:38:20,367 - [INFO] - <--------------- Entity Hiararchy statistics ----------------->
	2021-10-04 01:38:20,568 - [INFO] - Level 0 contains 1 types and 3415 triples
	2021-10-04 01:38:20,887 - [INFO] - Level 1 contains 66 types and 2006 triples
	2021-10-04 01:38:20,900 - [INFO] - Level 2 contains 136 types and 4273 triples
	2021-10-04 01:38:20,913 - [INFO] - Level 3 contains 213 types and 3560 triples
	2021-10-04 01:38:20,923 - [INFO] - Level 4 contains 262 types and 3369 triples


## Run Tests (pykeen models) 

### Symmetry behavioral tests on distmult or rotate:

	python tests/run.py --dataset FB15K237 --mode test --model_name rotate
    
The output will be printed as shown below, and will also be available in the results folder under dataset/symmetry:

	2021-10-04 14:00:57,100 - [INFO] - Starting test1 with rotate model
	2021-10-04 14:03:23,249 - [INFO] - On test1, MR: 1.2407678244972578, MRR: 0.9400152688974949, Hits@1: 0.9014624953269958, Hits@5: 0.988482654094696, Hits@10: 0.9965264797210693
	2021-10-04 14:03:23,249 - [INFO] - Starting test2 with rotate model
	2021-10-04 14:04:15,614 - [INFO] - On test2, MR: 23.446483180428135, MRR: 0.4409348919640765, Hits@1: 0.30351680517196655, Hits@5: 0.5894495248794556, Hits@10: 0.7025994062423706
	2021-10-04 14:04:15,614 - [INFO] - Starting test3 with rotate model
	2021-10-04 14:04:25,364 - [INFO] - On test3, MR: 1018.9469026548672, MRR: 0.04786047740344238, Hits@1: 0.008849557489156723, Hits@5: 0.06194690242409706, Hits@10: 0.12389380484819412
	2021-10-04 14:04:25,365 - [INFO] - Starting test4 with rotate model
	2021-10-04 14:05:38,900 - [INFO] - On test4, MR: 4901.459, MRR: 0.07606098649786266, Hits@1: 0.9496666789054871, Hits@5: 0.893666684627533, Hits@10: 0.8823333382606506

### Hierarchy behavioral tests on distmult or rotate:

       python tests/run.py --dataset FB15K237 --mode test --capability hierarchy --model_name rotate

### Run Tests on other models and other frameworks 

(To be added)
