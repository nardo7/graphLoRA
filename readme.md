# Evaluating fine-tuning graphormer with LoRA

For the course "Seminar Geometry and Topology in Deep Learning" from the Swiss Joint Master of Science in Computer Science at the University of Fribourg, I did a project whose main goal was to evaluate [LoRA](https://arxiv.org/pdf/2106.09685) on the [Graphormer](https://graphormer.readthedocs.io/en/latest/).

## Limitation

The [Graphormer](https://graphormer.readthedocs.io/en/latest/) is a foundational model pretrained in a large graph dataset called [OGB-LSC-PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/). This model can be then used for fine-tuning in downstream tasks such as graph classification. However, fine-tuning the whole model requires a lot of computational resources (mainly GPU memory) and time.

## LoRA

[LoRA](https://arxiv.org/pdf/2106.09685) is a parameter-efficient fine-tuning method that consists in freezing the pretrained model weights and injecting trainable rank-decomposition matrices into each layer of the Transformer architecture. This method has been shown to be very effective in Natural Language Processing (NLP) tasks, reducing the number of trainable parameters while achieving comparable performance to full fine-tuning. It can reduce the number of trainable parameters 10000 times (e.g., GPT-3 with 175B parameters can be fine-tuned with only 7M parameters using LoRA).

## Project

The main goal of this project is to evaluate the performance of LoRA when fine-tuning the Graphormer model in graph classification tasks. The experiments were conducted on the [MOLHIV](https://ogb.stanford.edu/docs/graphprop/#molhiv) dataset from the [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/). The performance of LoRA was compared to full fine-tuning in terms of accuracy, ROC-AUC, training time.

## Project Structure

- checkpointing.py: Code for saving and loading model checkpoints during training.
- lora.py: Own implementation of the LoRA method for parameter-efficient fine-tuning.
- optim.py: Code for defining optimizers and learning rate schedulers.
- simple_trainer.py: Own implementation of a simple trainer for training and evaluating models.
- trainer.py: Code for defining a Trainer class that handles that overrides the evaluation of models using the Hugging Face Trainer API for making it more efficient, improving run time by 3x.
- graphormer: extension of the Graphormer model to include a model weights regularization loss during fine-tuning.
- experiments:
  - graphormer-ft.py: Code for full fine-tuning of the Graphormer model using the Hugging Face Trainer API.
  - graphormer_lora.py: Code for fine-tuning the Graphormer model with LoRA using the Hugging Face Trainer API.
  - simple_run.py: Code for running quick experiments using the simple trainer.
    graph
  - graph_classification_regularized.py: Code for fine-tuning the Graphormer
    model with model weights regularization using the simple trainer.
- results.ipynb: Notebook for plotting and analyzing the results of the experiments.

## How to run the code

1. Clone this repository.
2. Install the required packages listed using `conda`
   ```bash
   conda env create -f environment.yml
   conda activate graphormer_lora
   ```
3. Run all the experiments
   ```bash
   python ./{experiment_file}.py
   ```
4. Run and analyze the results
   ```bash
   jupyter notebook results.ipynb
   ```

I also tried to implement [FLAG](https://arxiv.org/pdf/2010.09891) for adversarial training, which is very important for molecular datasets such as MOLHIV, but I could not finish it due to time constraints. I implemented the flag inner loop but I also had to integrate it into the model and this is the part I could not finish.
That's why the results are so poorly compared to the ones in the paper, i.e. just take as base the full fine-tuning results and the compare them with the other experiments.
