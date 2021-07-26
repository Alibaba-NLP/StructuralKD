# StructuralKD

The code is for our ACL-IJCNLP 2021 paper: [Structural Knowledge Distillation: Tractably Distilling Information for Structured Predictor](https://arxiv.org/abs/2010.05010)

StructuralKD is a framework for training stronger and smaller models through knowledge distillation (KD). StructuralKD can exactly calculate the KL divergence between different output structures between teacher and student models. 

## Guide

- [Requirements](#requirements)
- [Datasets](#datasets)
- [Scenarios](#scenarios)
  - [Teacher and Student Share the Same Factorization Form](#teacher-and-student-share-the-same-factorization-form)
    - [Linear-Chain CRF⇒Linear-ChainCRF](#linear-chain-crflinear-chaincrf)
      - [Teacher Models](#teacher-models)
      - [Training Student Models](#training-student-models)
    - [Graph-based Dependency Parsing⇒Dependency Parsing as Sequence Labeling](#graph-based-dependency-parsingdependency-parsing-as-sequence-labeling)
      - Coming Soon
  - [Student Factorization Produces More Fine-grained Substructures than Teacher Factorization](#student-factorization-produces-more-fine-grained-substructures-than-teacher-factorization)
    - [Linear-Chain CRF⇒MaxEnt](#linear-chain-crfmaxent)
      - [Teacher Models](#teacher-models-1)
      - [Training Student Models](#training-student-models-1)
    - [Second-Order Dependency Parsing⇒Dependency Parsing as Sequence Labeling](#second-order-dependency-parsingdependency-parsing-as-sequence-labeling)
      - Coming Soon
  - [Teacher Factorization Produces More Fine-grained Substructures than Student Factorization](#teacher-factorization-produces-more-fine-grained-substructures-than-student-factorization)
    - [MaxEnt⇒Linear-Chain CRF](#maxentlinear-chain-crf)
      - [Teacher Model](#teacher-model)
      - [Training Student Models](#training-student-models-2)
  - [Factorization Forms From Teacher andStudent are Incompatible](#factorization-forms-from-teacher-and-student-are-incompatible)
    - Coming Soon
- [Parse files](#parse-files)
- [Config File](#Config-File)
- [Citing Us](#Citing-Us)
- [Contact](#contact)

## Requirements
The project is based on PyTorch 1.1+ and Python 3.6+. To run our code, install:

```
pip install -r requirements.txt
```

The following requirements should be satisfied:
* [transformers](https://github.com/huggingface/transformers): **3.0.0** 

## Datasets
The datasets used in our paper are available [here](https://drive.google.com/drive/folders/1DFmz9KMJS6epm3TAMtL7PNG7IQV_JSAU?usp=sharing).

## Scenarios
Following the paper, we show how to apply StructuralKD in four scenarios.

### Teacher and Student Share the Same Factorization Form

#### Linear-Chain CRF⇒Linear-ChainCRF 

##### Teacher Models

We follow our previous work to train the CoNLL named entity recognition (NER) teachers. The teachers are available on [google drive](https://drive.google.com/drive/folders/1DFmz9KMJS6epm3TAMtL7PNG7IQV_JSAU?usp=sharing). Put these models in `resources/taggers`. 

An alternative way is training the teacher models by yourself: 
```bash
python train_with_teacher.py --config config/multi_bert_origflair_300epoch_2000batch_0.1lr_256hidden_de_monolingual_crf_sentloss_10patience_baseline_nodev_ner0.yaml #German
python train_with_teacher.py --config config/multi_bert_origflair_300epoch_2000batch_0.1lr_256hidden_en_monolingual_crf_sentloss_10patience_baseline_nodev_ner0.yaml #English
python train_with_teacher.py --config config/multi_bert_origflair_300epoch_2000batch_0.1lr_256hidden_es_monolingual_crf_sentloss_10patience_baseline_nodev_ner1.yaml #Spanish
python train_with_teacher.py --config config/multi_bert_origflair_300epoch_2000batch_0.1lr_256hidden_nl_monolingual_crf_sentloss_10patience_baseline_nodev_ner1.yaml #Dutch
```


##### Training Student Models

Run:
```bash
python train_with_teacher.py --config config/en_crf_ner.yaml #English
python train_with_teacher.py --config config/de_crf_ner.yaml #German
python train_with_teacher.py --config config/nl_crf_ner.yaml #Dutch
python train_with_teacher.py --config config/es_crf_ner.yaml #Spanish
```


#### Graph-based Dependency Parsing⇒Dependency Parsing as Sequence Labeling

The code is aviable in the branch DepKD. We follow the biaffine parser to train the graph-based dependency parser. The teachers are available on [google drive](https://drive.google.com/drive/folders/1DFmz9KMJS6epm3TAMtL7PNG7IQV_JSAU?usp=sharing). Put these models in `resources/taggers`. 

An alternative way is training the teacher models by yourself: 
```bash
python train_with_teacher.py --config config/word_char_500epoch_0.5inter_5000batch_0.002lr_400hidden_ptb_monolingual_nocrf_fast_freeze_nodev_dependency30.yaml
```

##### Training Student Models

Run:
```bash
python train_with_teacher.py --config config/ptb-dp_as_sl-1st.yaml
```

---


### Student Factorization Produces More Fine-grained Substructures than Teacher Factorization

#### Linear-Chain CRF⇒MaxEnt

##### Teacher Models

The teacher models are identical to the models in [Linear-Chain CRF⇒Linear-ChainCRF]().

##### Training Student Models

Run:
```bash
python train_with_teacher.py --config config/en_maxent_ner.yaml #English
python train_with_teacher.py --config config/de_maxent_ner.yaml #German
python train_with_teacher.py --config config/nl_maxent_ner.yaml #Dutch
python train_with_teacher.py --config config/es_maxent_ner.yaml #Spanish
```



#### Second-Order Dependency Parsing⇒Dependency Parsing as Sequence Labeling
The code is aviable in the branch DepKD. We follow our previous model to train the graph-based second-order dependency parser. The teachers are available on [google drive](https://drive.google.com/drive/folders/1DFmz9KMJS6epm3TAMtL7PNG7IQV_JSAU?usp=sharing). Put these models in `resources/taggers`. 

An alternative way is training the teacher models by yourself: 
```bash
python train_with_teacher.py --config config/word_char_500epoch_0.5inter_5000batch_0.002lr_400hidden_ptb_monolingual_2nd_nocrf_fast_freeze_nodev_dependency30.yaml
```

##### Training Student Models

Run:
```bash
python train_with_teacher.py --config config/ptb-dp_as_sl-2nd.yaml
```

---

### Teacher Factorization Produces More Fine-grained Substructures than Student Factorization

#### MaxEnt⇒Linear-Chain CRF

##### Teacher Model

The teacher is a multilingual teacher trained on WikiAnn datasets with four languages (Dutch, English, German, Spanish) The teacher is available on [google drive](https://drive.google.com/drive/folders/1DFmz9KMJS6epm3TAMtL7PNG7IQV_JSAU?usp=sharing). Put these models in `resources/taggers`. 

Similarly, the teacher model can be trained by yourself: 
```
python train_with_teacher.py --config config/multi-bert_10epoch_32batch_0.00005lr_10000lrrate_multilingual_nocrf_fast_relearn_sentbatch_sentloss_finetune_conlllang_nodev_panx_ner9.yaml
```

##### Training Student Models

In this case, we train the student model for four zero shot langauges, i.e. Basque, Hebrew, Persian and Tamil.

For each language, run:
```
python train_with_teacher.py --config config/ta_ner.yaml
python train_with_teacher.py --config config/fa_ner.yaml
python train_with_teacher.py --config config/eu_ner.yaml
python train_with_teacher.py --config config/he_ner.yaml
```

---

### Factorization Forms From Teacher and Student are Incompatible

Coming Soon ...

---

### Train on Your Own Dataset

To set the dataset manully, you can set the dataset in the `$config_file` by:

```yaml
targets: ner
ner:
  Corpus: ColumnCorpus-1
  ColumnCorpus-1: 
    data_folder: datasets/conll_03_english
    column_format:
      0: text
      1: pos
      2: chunk
      3: ner
    tag_to_bioes: ner
  tag_dictionary: resources/taggers/your_ner_tags.pkl
```


The `tag_dictionary` is a path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically. The dataset format is: `Corpus: $CorpusClassName-$id`, where `$id` is the name of datasets (anything you like). You can train multiple datasets jointly. For example:

Please refer to [Config File](#Config-File) for more details.

## Parse files

If you want to parse a certain file, add `train` in the file name and put the file in a certain `$dir` (for example, `parse_file_dir/train.your_file_name`). Run:

```
CUDA_VISIBLE_DEVICES=0 python train.py --config $config_file --parse --target_dir $dir --keep_order
```

The format of the file should be `column_format={0: 'text', 1:'ner'}` for sequence labeling or you can modifiy line 232 in `train.py`. The parsed results will be in `outputs/`.
Note that you may need to preprocess your file with the dummy tags for prediction, please check this [issue](https://github.com/Alibaba-NLP/ACE/issues/12) for more details.

## Config File

The config files are based on yaml format.

* `targets`: The target task
  * `ner`: named entity recognition
  * `upos`: part-of-speech tagging
  * `chunk`: chunking
  * `ast`: abstract extraction
  * `dependency`: dependency parsing
  * `enhancedud`: semantic dependency parsing/enhanced universal dependency parsing
* `ner`: An example for the `targets`. If `targets: ner`, then the code will read the values with the key of `ner`.
  * `Corpus`: The training corpora for the model, use `:` to split different corpora.
  * `tag_dictionary`: A path to the tag dictionary for the task. If the path does not exist, the code will generate a tag dictionary at the path automatically.
* `target_dir`: Save directory.
* `model_name`: The trained models will be save in `$target_dir/$model_name`.
* `model`: The model to train, depending on the task.
  * `FastSequenceTagger`: Sequence labeling model. The values are the parameters.
  * `SemanticDependencyParser`: Syntactic/semantic dependency parsing model. The values are the parameters.
* `embeddings`: The embeddings for the model, each key is the class name of the embedding and the values of the key are the parameters, see `flair/embeddings.py` for more details. For each embedding, use `$classname-$id` to represent the class. For example, if you want to use BERT and M-BERT for a single model, you can name: `TransformerWordEmbeddings-0`, `TransformerWordEmbeddings-1`.
* `trainer`: The trainer class.
  * `ModelFinetuner`: The trainer for fine-tuning embeddings or simply train a task model without ACE.
  * `ReinforcementTrainer`: The trainer for training ACE.
* `train`: the parameters for the `train` function in `trainer` (for example, `ReinforcementTrainer.train()`).


## Citing Us
If you feel the code helpful, please cite:
```
@inproceedings{wang2021improving,
    title = "{{Improving Named Entity Recognition by External Context Retrieving and Cooperative Learning}}",
    author={Wang, Xinyu and Jiang, Yong and Bach, Nguyen and Wang, Tao and Huang, Zhongqiang and Huang, Fei and Tu, Kewei},
    booktitle = "{the Joint Conference of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (\textbf{ACL-IJCNLP 2021})}",
    month = aug,
    year = "2021",
    publisher = "Association for Computational Linguistics",
}
```

## Contact 

Feel free to email your questions or comments to issues or to [Xinyu Wang](http://wangxinyu0922.github.io/).

