# Evaluating fine-tuning Graphormer with LoRA

For the course "Seminar Geometry and Topology in Deep Learning" from the Swiss Joint Master of Science in Computer Science at the University of Fribourg, I did a project whose main goal was to evaluate [LoRA](https://arxiv.org/pdf/2106.09685) on the [Graphormer](https://graphormer.readthedocs.io/en/latest/).

## Limitation

The [Graphormer](https://graphormer.readthedocs.io/en/latest/) is a foundational model pretrained in a large graph dataset called [OGB-LSC-PCQM4Mv2](https://ogb.stanford.edu/docs/lsc/pcqm4mv2/). This model can be then used for fine-tuning in downstream tasks such as graph classification. However, fine-tuning the whole model requires a lot of computational resources (mainly GPU memory) and time.

## LoRA

[LoRA](https://arxiv.org/pdf/2106.09685) is a parameter-efficient fine-tuning method that consists in freezing the pretrained model weights and injecting trainable rank-decomposition matrices into each layer of the Transformer architecture. This method has been shown to be very effective in Natural Language Processing (NLP) tasks, reducing the number of trainable parameters while achieving comparable performance to full fine-tuning. It can reduce the number of trainable parameters 10000 times (e.g., GPT-3 with 175B parameters can be fine-tuned with only 7M parameters using LoRA).

## Project

The main goal of this project is to evaluate the performance of LoRA when fine-tuning the Graphormer model in graph classification tasks. The experiments were conducted on the [MOLHIV](https://ogb.stanford.edu/docs/graphprop/#molhiv) dataset from the [Open Graph Benchmark (OGB)](https://ogb.stanford.edu/). The performance of LoRA was compared to full fine-tuning in terms of accuracy, ROC-AUC, training time.

## Results

Please check the results notebook [results.ipynb](results.ipynb) for more details about the results obtained in the experiments.

## Project Structure

Please check the following structure of the project. Here you can concretely see what I did and where the experiments are located and the results, so you can check my work:

- checkpointing.py: Code for saving and loading model checkpoints during training.
- lora.py: Own implementation of the LoRA method for parameter-efficient fine-tuning.
- optim.py: Code for defining optimizers and learning rate schedulers.
- simple_trainer.py: Own implementation of a simple trainer for training and evaluating models.
- trainer.py: Code for defining a Trainer class that overrides the evaluation of models using the Hugging Face Trainer API for making it more efficient, improving run time by 3x compared to the HF implementation.
- graphormer: Extension of the Graphormer model to include a model weights regularization loss during fine-tuning.
- experiments:
  - graphormer-ft.py: Code for full fine-tuning of the Graphormer model using the Hugging Face Trainer API.
  - graphormer_lora.py: Code for fine-tuning the Graphormer model with LoRA using the Hugging Face Trainer API.
  - simple_train_run.py: Code for running quick experiments using the simple trainer. You need to set the variable `apply_lora` to True or False depending on whether you want to use LoRA or not.
  - graph_classification_regularized.py: Code for fine-tuning the Graphormer
    model with model weights regularization using the simple trainer.
- results.ipynb: Notebook for plotting and analyzing the results of the experiments.

## How to run the code

### Dataset

You need to download the MOLHIV dataset from the [OGB website](https://ogb.stanford.edu/docs/graphprop/#molhiv) and place it in the `./data` folder.

### Running the experiments

1. Clone this repository.
2. Install the required packages listed using `conda`
   ```bash
   conda env create -f environment.yml
   conda activate graphormer_lora
   ```
3. Run all the experiments
   ```bash
   python ./simple_train_run.py
   ```
4. Alternatively, you can run each experiment separately:
   - Full fine-tuning
     ```bash
     python ./experiments/graphormer-ft.py
     ```
   - LoRA fine-tuning
     ```bash
     python ./experiments/graphormer_lora.py
     ```
   - Fine-tuning with model weights regularization
     ```bash
     python ./experiments/graph_classification_regularized.py
     ```
5. Run and analyze the results
   ```bash
   jupyter notebook results.ipynb
   ```
   **Important Note**: In the simple_train_run.py file, you need to set the variable `apply_lora` to True or False depending on whether you want to use LoRA or not. Also, change the lr parameter accordingly, since LoRA usually requires a higher learning rate than full fine-tuning.

I also tried to implement [FLAG](https://arxiv.org/pdf/2010.09891) for adversarial training, which is very important for molecular datasets such as MOLHIV, but I could not finish it due to time constraints. I implemented the flag inner loop (see simple_trainer.py) but I also had to integrate it into the model and this is the part I could not finish.
That's why the results are so poorly compared to the ones in the paper, i.e. just take as base the full fine-tuning results and the compare them with the other experiments.

### Hyperparameters

The hyperparameters used for the experiments were the default ones given by the paper for the MOLHIV dataset. For Lora, I used r=16 as the rank for the decomposition matrices, which is a common value used in the literature for this task. This gave a reduction ratio of ~99% of trainable parameters compared to full fine-tuning and improved the training time by 3x, making one epoch taking around 20 minutes instead of 1 hour.

### Setup

All the experiments were run in a M1 MacBook Pro with 32GB of RAM (CPU & GPU).

## Acknowledgements

I would like to thank Johannes Schmidt for his promptly support and guidance throughout the project. All this work is completely based on the original [Graphormer implementation](https://github.com/microsoft/Graphormer) and the [Hugging Face Transformers](https://huggingface.co/microsoft/Graphormer) library.

## References

- [Do Transformers Really Perform Bad for Graph Representation?](https://arxiv.org/abs/2106.05234)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/pdf/2106.09685)
- [Robust Optimization as Data Augmentation for Large-scale Graphs](https://arxiv.org/pdf/2010.09891)
- [Attention is All You Need](https://arxiv.org/abs/1706.03762)
