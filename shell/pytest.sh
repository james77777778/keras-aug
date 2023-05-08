#!/bin/bash
pytest --durations=10 --cov=keras_aug --cov-report=term:skip-covered --cov-report=xml:coverage.xml
