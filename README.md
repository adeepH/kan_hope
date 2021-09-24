# KanHope
This is the code for the paper "Hope Speech detection in under-resourced Kannada language"

This is a collaborative work by [Adeep Hande](https://github.com/adeepH), Ruba Priyadharshini, Anbukkarasi Sampath, Kingston Pal Thamburaj, [Prabakaran Chandran](https://github.com/Prabakaran-Chandran), and [Bharathi Raja Chakravarthi](https://github.com/bharathichezhiyan)

## [Steps to run the Vanilla sequence classification tasks:](https://github.com/adeepH/KanHope/tree/main/Vanilla%20sequence%20Classification)
1) Download the corresponding files from Zenodo:
```buildoutcfg
https://zenodo.org/record/5006517/
```
2) Set the path to `'path_to_repo/KanHope/Dual Channel models/'`.

3) For the models that follow the architecture of BERT, run the `classifier.py` and find the string `'read_csv'`. Add the paths to the train, test, and validation dataframes. Change the path to the dataset where the files have been stored after downloading from [Zenodo](https://zenodo.org/record/4904729/).

4) Run `test.py` for inference.

5) Under the same directory, run `get_predictions.py` to view classification reports and confusion matrix.
## [Steps to run the Dual Channel BERT-based models (DC-BERT4HOPE)](https://github.com/adeepH/KanHope/tree/main/Dual%20Channel%20models)
1) Download the English translations of the code-mixed Kannada-English dataset, along with the splits:
```buildoutcfg
https://Zenodo.org/record/4904729/
```
2) run `dc_classifier.py` to train the Dual channel BERT model.

3) For the names of the models (`model1;model2`), follow the naming conventions as listed in [Huggingface Transformers' pretrained models](https://huggingface.co/transformers/pretrained_models.html).
   a)`model1:` Monolingual English language model (Translated Texts).
   b)`model2:` Multilingual language model (Kannada-English code-mixed text).
   
4) under the same directory run `get_predictions.py` to view the classification reports and confusion matrix.

5) The architecture of the dual channel model is as follows:

<img width = "60%" src = "https://github.com/adeepH/KanHope/blob/main/Images/dc_bert4hope.png">

This approach could be used for any multilingual datasets. The weights of the fine-tuned models are available on my Huggingface account [AdWeeb](https://huggingface.co/AdWeeb).

We have provided the [notebooks](https://github.com/adeepH/KanHope/tree/main/Notebooks) for reference.

# Experiments, Results, and Discussions
The code and their explanation for all the experiments are present in the Jupyter Notebook. We document interesting findings, results, discussions and qualitative analysis in the manuscript.

<img width= "60%" src = "https://github.com/adeepH/KanHope/blob/main/Images/results.png">

If you use our dataset, and/or find our codes useful, please cite our paper:
```buildoutcfg
@misc{hande2021hope,
      title={Hope Speech detection in under-resourced Kannada language}, 
      author={Adeep Hande and Ruba Priyadharshini and Anbukkarasi Sampath and Kingston Pal Thamburaj and Prabakaran Chandran and Bharathi Raja Chakravarthi},
      year={2021},
      eprint={2108.04616},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```