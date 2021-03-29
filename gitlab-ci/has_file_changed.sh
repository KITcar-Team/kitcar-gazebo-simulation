#!/bin/bash

git diff --quiet "$1" 2> /dev/null
echo $?
