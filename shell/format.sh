#!/bin/bash
ruff check --fix --show-fixes .
black .
