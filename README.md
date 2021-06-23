# KanHope
This is the code for the paper "Hope Speech detection in under-resourced Kannada language"

This is a collaborative work by [Adeep Hande](https://github.com/adeepH), Ruba Priyadharshini, Anbukkarasi Sampath, Kingston Pal Thamburaj, Prabakaran Chandran, and [Bharathi Raja Chakravarthi](https://github.com/bharathichezhiyan)

## [Steps to run the Vanilla sequence classification tasks:](https://github.com/adeepH/KanHope/tree/main/Vanilla%20sequence%20Classification)
1) Download the corresponding files from Zenodo:
```buildoutcfg
https://zenodo.org/record/5006517/
```
2) For the models that follow the architecture of BERT, run the `classifier.py` and find the string `'read_csv'`. Change the path to the dataset where the files have been stored after downloading from [Zenodo](https://zenodo.org/record/4904729/).

3) Run `test.py` for inference.

## [Steps to run the Dual Channel BERT-based models (DC-BERT4HOPE)](https://github.com/adeepH/KanHope/tree/main/Dual%20Channel%20models)
1) Download the English translations of the code-mixed Kannada-English dataset, along with the splits:
```buildoutcfg
https://Zenodo.org/record/4904729/
```
2) run `dc_classifier.py` to train the Dual channel BERT model.

3) For the names of the models (`model1;model2`), follow the naming conventions as listed in [Huggingface Transformers' pretrained models](https://huggingface.co/transformers/pretrained_models.html).
   a)`model1:` Monolingual English language model (Translated Texts).
   b)`model2:` Multilingual language model (Kannada-English code-mixed text).
   

<img width = "2406" src = "https://github.com/SiddhanthHegde/You-Need-to-Pay-More-Attention/blob/main/Meme%20img%20.jpg">