#!/bin/bash -e
docker build . --file Dockerfile --tag trial
docker run --rm -it -v "$(pwd)":/models trial python3 -m pytest ivy_models_tests/