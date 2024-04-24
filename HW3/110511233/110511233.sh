#! /bin/bash

# create venv and install dependencies
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# the dataset should be on the same folder
# and .env file contains 'HUGGINGFACE_TOKEN=...'
.venv/bin/python t5-xl.py
.venv/bin/python generate_result.py
