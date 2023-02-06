# Adversarial attack transfer

Authors: Jérémie Dentan, Nathanaël Cuvelle--Magar, Abdellah El Mrini

This repository implements the main experimentations of *Task-generalizable Adversarial Attack based on Perceptual Metric* by Muzammal Naseer, Salman H. Khan, Shafin Rahman, Fatih Porikli in 2019. You can find this paper [here](https://arxiv.org/abs/1811.09020).

## Main results

Our main results are presented in the following table. It presents the score of several classifiers on the [Tiny ImageNet-200](https://paperswithcode.com/dataset/tiny-imagenet) dataset, depending on the attack previously performed on those images. The left columns denotes the attack performed on the images: `original` are the images without attack; `vgg_conv{layer}_{nb}steps` are the NRDM attacks (for description, cf. the paper of Naseer et al.) using VGG16 network at layer `layer` with `nb` step for the attack; and `decorrelate_FFT_{layer}_{nb}steps` are the attacks in the Fourrier space (cf. script `src/attack/decorrelate_fft.py`) with using layer number `layer` and `nb` steps.

|Attack|Accuracy of DenseNet|Accuracy of ResNet|Accuracy of VGG|
|---|:---:|:---:|:---:|
original|0.6392|0.7388|0.4075
vgg_conv22_3steps|0.2567|0.2885|0.1367
vgg_conv22_5steps|0.2346|0.2556|0.1112
vgg_conv22_7steps|0.2425|0.2481|0.1002
vgg_conv22_10steps|0.2456|0.2477|0.0979
vgg_conv33_3steps|0.2345|0.2605|0.0722
vgg_conv33_5steps|0.2061|0.2179|0.0543
vgg_conv33_7steps|0.199|0.2107|0.0475
vgg_conv33_10steps|**0.1957**|**0.2011**|0.045
vgg_conv43_3steps|0.2329|0.2417|0.0454
vgg_conv43_5steps|0.2192|0.2238|0.0326
vgg_conv43_7steps|0.207|0.2142|**0.0298**
vgg_conv43_10steps|0.2029|0.2068|0.0301
decorrelate_FFT_5_100steps|0.5591|0.6230|0.3610
decorrelate_FFT_5_250steps|0.4728|0.5463|0.3546
decorrelate_FFT_7_100steps|0.5654|0.6357|0.3769
decorrelate_FFT_7_250steps|0.5047|0.5303|0.3322

Our results are coherent with what in claimed in the paper of Naseer et al.: the best result comes when using layer `conv33` in VGG16, and we obtain very good transferability of the attacks.

For more details about our results, a report (in French) is available in the `/doc` folder.

## Run the code

### Set up your environment

This code is meant to run in **Python 3.8** with the PYTHONPATH set to the root of the project. We advise you to use [Python native virtual environments](https://docs.python.org/3/library/venv.html) or [Conda virtual environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). To do so, run the following from the root of the repository:

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

### Execute the code

Two steps are implemented in this repository: The first is computing adversarial attacks on images using either the NRDM algorithm at nrmd.py module, or the decorrelate_fft_attack.py module. The second step is evaluating those attacks on different models (not necessarily those the threat models were trained on). The dataset we use is [Tiny_ImageNet-200](https://paperswithcode.com/dataset/tiny-imagenet). It is automatically downloaded by our pipeline.

**To run the attacks:**

To run the NRDM attack as described in *Task-generalizable Adversarial Attack based on Perceptual Metric*, execute the following line. This will create tensors corresponding to adversarial images in `/output/corruptions`, as well as logs in `/logs`.

```bash
python -m src.attack
```

Two types of attacks are run: regular NRDM attacks as described in the paper, and NRDM attacks on the Fourrier space.

**To run the evaluation:**

To run the evaluation of the attacks, execute the following line. The output of this step will be saved in `/output/classifiers`. There will be pretrained classifiers, as well as a `.csv` result file for each classifier corresponding to its performances on the adversarial images computed at the previous step. Moreover, there will be some logs in `/logs`.

```bash
python -m src.classifiers_evaluation
```

## Precomputed corruptions and pretrained classifiers

The computation of adversarial images and the training of the models takes time. More precisely:

* On CPU, the computation of the adversarial images in regular space takes about 1h for each attack, and there are 12 of them.
* On GPU, the computation of the adversarial images in regular space takes less than 3min per attack.
* On GPU, the computation of the adversarial images in Fourrier space takes about 20min per attack, and there are 4 of them.
* On GPU, the training of the models takes less than 5min by model, and there are 3 of them.
* On GPU, the evaluation with a trained model takes of one type of attack takes about 10sec, and there are 12 regular NRDM attacks and 4 attacks on the Fourrier space.

To run our code with batch size of 32, you will need at least 3Go of graphic memory. It you don't have that, please consider reducing the batch size in `src/utils/datasets`.

However, you can directly use the adversariales images and models we generated / trained. To do so:

* Download the adversarial images [here](https://www.icloud.com/iclouddrive/0064OPlcGKoM2EXoaMuCHz1kw#corruptions) and put the `corruptions` folder in `/output`.
* Download the pretrained models [here](https://www.icloud.com/iclouddrive/0b4D60mp4V7Lw_cYQIklMwSAw#classifiers) and put the `classifiers` folder in `/output`.

## License and Disclaimer

You may use this software under the Apache 2.0 License. See LICENSE.
