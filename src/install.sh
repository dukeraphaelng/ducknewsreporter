#!/bin/bash

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 -m textblob.download_corpora
python3 -m spacy download en_core_web_sm
