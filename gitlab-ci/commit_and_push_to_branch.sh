#!/bin/bash

MSG=$1
BRANCH=$2

git checkout -b $BRANCH
git commit -m "$MSG"
git push --set-upstream origin $BRANCH
