#!/bin/sh

/vol/bitbucket/xg1020/nlp-tone-detect/.venv/bin/pip install -r /vol/bitbucket/xg1020/nlp-tone-detect/requirements.txt

. /vol/cuda/12.0.0/setup.sh
/vol/bitbucket/xg1020/nlp-tone-detect/.venv/bin/python /vol/bitbucket/xg1020/nlp-tone-detect/pipeline.py
