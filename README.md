# Poisoning-Semi-Supervised-Adversarial-Training
Trustworthy ML final project
SVHN dataset:
73257 digits for training, 26032 digits for testing, and 531131 additional, somewhat less difficult samples, to use as extra training data
http://ufldl.stanford.edu/housenumbers/

The Street View House Numbers (SVHN) dataset is composed of 10 distinct digit classes and is
organized into three primary segments:
1.Training Dataset:
consists of approximately 73,000 samples.
2.Extra Dataset:
with around 531,000 samples. In our approach, this dataset is initially used as unlabeled data, with labels removed, and subsequently employed in the self-training process.
3.Test Dataset: Comprising roughly 26,000 samples, the test dataset is reserved for evaluating
the model’s performance

Dependencies:
Before running the code, make sure you have all the required dependencies installed. You can install the necessary Python packages using `pip`:
pip install -r requirements.txt


Note: code tested on 2 GPUs in parallel, each with 12GB of memory. Running on CPUs or GPUs with less memory might require adjustments.


Self Training - Psuedo Labels:
-Training: We initiate the model’s training using only the labeled samples from the "train" dataset.
-Self-Training: After the initial training, we deploy the pre-trained model to predict labels for
the "extra" dataset, thereby generating ’pseudo-labels’ for these unlabeled samples.
Combined Training: To further enhance the model’s capabilities, we conduct training using
both the original labeled "train" dataset and the "extra" dataset, now augmented with
pseudo-labels.
-Testing and Evaluation: Finally, we evaluate the model’s performance by testing it on a separate test dataset and measuring its accuracy in classifying benign examples.

You can run the code with default parameters as follows:
python self_training.py --model resnet_16_8 --model_dir /path/to/your/model/checkpoint \
                          --model_epoch 200 --batch_size 128 \
                          --data_dir data/ \
                          --data_filename ti_top_50000_pred_v3.1.pickle \
                          --output_dir data/ \
                          --output_filename pseudolabeled-top50k.pickle


The Backdoor Attack on the SVHN network has two main steps:
Initial Training: The model is trained on labeled data.
Self-Training: The trained model labels unlabeled data, creating 'pseudo-labels.'
The attack's goals are:
Backdoor Trigger: Inserting a single pixel or pattern into images.
Backdoor Target Class Association: Changing labels of specific digits to other digits.

python evaluate_backdoored_model.py.py --model resnet_16_8 --model_dir /path/to/your/model/checkpoint \
                          --model_epoch 200 --batch_size 10 \
                          --data_dir data/ \
                          --data_filename ti_top_50000_pred_v3.1.pickle \
                          --output_dir data/ \
                          --output_filename pseudolabeled-top50k.pickle

Here's an explanation of the command-line arguments:
--model: Specify the name of the model (e.g., resnet_16_8).
--model_dir: Provide the path to the directory containing the trained model checkpoint.
--model_epoch: Specify the number of epochs the model was trained for.
--batch_size: Set the batch size for data processing.
--data_dir: Specify the directory where the unlabeled data is located.
--data_filename: Specify the filename of the file with unlabeled data.
--output_dir: Set the directory where the output will be saved.
--output_filename: Specify the filename for the saved output.


other files:
-atacks.py:
Backdoor Attack Implementation
This script contains the implementation of a backdoor attack, including methods
for generating poisoned images and poisoning datasets.
-dataloader.py
-dataset.py:
This script defines a custom dataset class, `SemiSupervisedSVHN`, which inherits from
`torchvision.datasets.SVHN`. It allows for semi-supervised learning on the SVHN dataset,
including the option to use pseudo-labels for unlabeled data.
-models.py:This script defines various neural network models including fully connected networks (NN)
-utils.py:This script provides various utility functions for data analysis, plotting, and training.
