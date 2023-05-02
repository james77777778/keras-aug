#!/bin/bash
pytest --durations=10 --cov-report=html:htmlcov --cov-report=term:skip-covered --cov=./
