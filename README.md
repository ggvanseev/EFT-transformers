# EFT-transformers
This is an implementation of the effective field theory implementation at leading order of transformers, currently only the self attention block

## Setup

1. **Cloning the repository:**

   ```bash
   git clone https://github.com/pabloiyu/Masters-Project.git

2. **Create a Conda environment:**

   ```bash
   conda create --name my-project-env python=3.12 
   conda activate my-project-env

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt 

## Usage | Running Files

In order to run any file, it is sufficient to simply type the following into terminal:

   ```bash
      python name_of_file.py
   ```

Within each file, there are a set of hyperparameters and additional settings that can be modified. Note that this at the moment also includes hardcoded paths. In the future, I want to delete those.

## Usage | At Initialization
**creation_n_analysis.py**

   * Computes and compares the analytical and numerical results for an MLP and a multi-head self-attention (MHSA) neural network. Note that the first layer of the MHSA is an MLP. Various (non-intuitive) settings are explained with comments at the top of the file. The general structure allows you to choose the index of the correlation function to calculate. You can use existing results or create new ones. When creating new results, the hyperparameters need to be set.
   
   If desired, read the added paper for more background. See section .. for the method and setup, while secion .. and .. should provide some theoretical background.