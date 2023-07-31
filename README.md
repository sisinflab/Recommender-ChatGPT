# ChatGPT as Zero-Shot Recommender Systems
This repository contains the source code and datasets for the paper titled "**Evaluating ChatGPT as a Recommender System: A Rigorous Approach**" which has been submitted to [UMUAI Journal](http://www.umuai.org/).

To correctly clone the repo, use:
```bash
git clone --recursive XYZ
```

The project includes the necessary source code and files to conduct experiments for the baselines, ChatGPT, GPT-3.5 and PaLM-2.
Before running these codes, ensure that you have Python version 3.8.6 or later installed on your device.
To set up the required environment, you can create a virtual environment and install the necessary dependencies using the provided requirements files with the following steps:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, you can use [**conda**](https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html) to recreate the same environment by executing the following command:
```bash
conda env create -f env/env_recommender-ChatGPT.yml
conda activate Recommender-ChatGPT
```

To correctly initialize Elliot, from ```../Recommender-ChatGPT/external/```:
```bash
pip install --upgrade pip
pip install -e . --verbose
```

To obtain the results, after inserting the API token, execute the following command:
```bash
python code/main.py
```
The results are stored in the _results_ folder.
