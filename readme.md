#### NER as Parsing => MaxEnt

##### Teacher Models
The code is available in the branch "case4". We follow previous work to train the named entity recognition(NER) teachers on CoNLL/WikiAnn datasets, the teachers are available [here](https://drive.google.com/drive/folders/1Q-GYL90sJ3T07CiimBTJIqKmDjo1zGcp?usp=sharing). Unzip the files to `./saves`

To train a teacher model by yourself, you can follow the example command as follow:
```bash
python train_with_teacher.py --config configs/conll_teachers/conll03_bs3500_lr1e-3_epoch1k1_flair_fastword_bert_de.yaml  
python train_with_teacher.py --config configs/wikiann_teachers/wikiann_bs3500_lr1e-3_epoch1k1_flair_fastword_bert_panx_en.yaml  
```
All config files are available in `configs/conll_teachers` and `configs/wikiann_teachers` 

##### Training Student Models

The example command to train a baseline student model:
```bash
python train_with_teacher.py --config configs/train_students/baselines/conll_baseline_de.yaml    # CoNLL datasets, German
python train_with_teacher.py --config configs/train_students/baselines/wikiann_baseline_en.yaml    # WikiAnn datasets, English
```
The example command to train a student model with structural knowledge distillation:
```bash
python train_with_teacher.py --config configs/train_students/kd/conll_kd_es.yaml # CoNLL datasets, Spanish
python train_with_teacher.py --config configs/train_students/kd/wikiann_kd_nl.yaml # WikiAnn datasets, Dutch
python train_with_teacher.py --config configs/train_students/kd/wikiann_3k_kd_de.yaml # WikiAnn datasets with 3k unlabeled sentences, German
```
All config files are availabel in `configs/train_students`