#!/bin/bash
pytest --durations=10 --cov=./ --cov-report=html:htmlcov --cov-report=term
