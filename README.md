# FewNERD LLM Prompting

[![Python 3.8](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-370/) [![](https://img.shields.io/github/license/lisskor/few-nerd-prompting.svg)](https://github.com/lisskor/few-nerd-prompting/blob/main/LICENSE)

This repository uses [Clarifai](https://www.clarifai.com/) to prompt an LLM and perform few-shot NER on the [Few-NERD dataset](https://aclanthology.org/2021.acl-long.248/) using methods described in the [GPT-NER](https://github.com/ShuheWang1998/GPT-NER) paper. 

## 🚀 Installation

**Requirements**

In a new virtual environment, install the required packages using
```
pip install -r requirements.txt
```

**Clarifai**

To use Clarifai, [sign up for a Free Account](https://clarifai.com/signup) or login then [get a Personal Access Token](https://docs.clarifai.com/clarifai-basics/authentication/personal-access-tokens/) (PAT).

**Few-NERD**

Download the [Few-NERD from the link](https://ningding97.github.io/fewnerd/).

## 💪 Usage
To use LLMs for few-shot NER, run
```
python prompt_llm.py --pat CLARIFAI_PAT --data_file NEW_NERD_EPISODE
```
