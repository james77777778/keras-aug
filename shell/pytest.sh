#!/bin/bash
pytest --durations=10 --cov=keras_aug --cov-report=html:htmlcov --cov-report=term:skip-covered --cov-report=xml:coverage.xml
