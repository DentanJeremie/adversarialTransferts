# Adversarial attack transfer

Authors: Jérémie Dentan, Nathanaël Cuvelle--Magar, Abdellah El Mrini

This repository implements the main experimentations of *Task-generalizable Adversarial Attack based on Perceptual Metric* by Muzammal Naseer, Salman H. Khan, Shafin Rahman, Fatih Porikli in 2019. You can find this paper [here](https://arxiv.org/abs/1811.09020).

## Run the code

### Set up your environment

This code is meant to run in **Python 3.8** with the PYTHONPATH set to the root of the project. We advise you to use [Python native virtual environments](https://docs.python.org/3/library/venv.html) or [Conda virtual environments](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). To do so, run the following from the root of the repository:

```bash
pip install -r requirements.txt
export PYTHONPATH=$(pwd)
```

### Execute the code

Two steps are implemented in this repository: computing adversarial attacks on images with NRDM algorithm, and evaluating thos attacks. The dataset we use is [Tiny_ImageNet-200](https://paperswithcode.com/dataset/tiny-imagenet). It is automatically downloaded by our pipeline.

**To run the attacks:**

```bash
python -m src.attack
```

This will create tensors corresponding to adversarial images in `/output/corruptions`, as well as logs in `/logs`.

**To run the evaluation:**

```bash
python -m src.classifiers_evaluation
```

The output of this step will be saved in `/output/classifiers`. There will be pretrained classifiers, as well as a `.csv` result file for each classifier corresponding to its performances on the adversarial images computed at the previous step. Moreover, there will be some logs in `/logs`.

## Precomputed corruptions and pretrained classifiers

The computation of adversarial images and the training of the models takes time. More precisely:

* On CPU, the computation of the adversarial images takes about 1h for each attack, and there are 15 of them.
* On GPU, the computatino of the adversarial images takes takes less than 1min per attack.
* On GPU, the training of the models takes less than 5min by model, and there are 3 of them.

To run our code with batch size of 32, you will need at least 3Go of graphic memory. It you don't have that, please consider reducing the batch size in `src/utils/datasets`.

However, you can directly use the adversariales images and models we generated / trained. To do so:

* Download the adversarial images [here](https://www.icloud.com/iclouddrive/0d6AmPWWqsORnrb5iqlD7VGNQ#corruptions) and put the `corruptions` folder in `/output`.
* Download the pretrained models [here](https://www.icloud.com/iclouddrive/0d0I13FmCcx7oL9mOwQcf2B-w#classifiers) and put the `classifiers` folder in `/output`.

## License and Disclaimer

You may use this software under the Apache 2.0 License. See LICENSE.
