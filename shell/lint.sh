#!/bin/bash
# Usage: # lint.sh can be used without arguments to lint the entire project:
#
# ./lint.sh
#
# or with arguments to lint a subset of files
#
# ./lint.sh examples/*

files="."
if [ $# -ne 0  ]
  then
    files=$@
fi

ruff check $files
if ! [ $? -eq 0 ]
then
  echo "Please fix the code style issue."
  exit 1
fi
[ $# -eq 0 ] && echo "no issues with ruff"

black --check $files
if ! [ $? -eq 0 ]
then
  echo "Please run \"sh shell/format.sh\" to format the code."
    exit 1
fi
[ $# -eq 0  ] && echo "no issues with black"
echo "linting success!"
