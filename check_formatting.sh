#!/bin/bash
# this is a git pre-commit hook that checks if source files included in the commit are correctly formatted

# don't check formatting if this is a merge-commit
# For how this check works look at this: https://stackoverflow.com/questions/27800512/bypass-pre-commit-hook-for-merge-commits
if git rev-parse --verify -q MERGE_HEAD
then
exit 0
fi

CLANG_FORMAT=$(which clang-format-3.6)

if [ "$CLANG_FORMAT" = "" ]
then
CLANG_FORMAT=$(which clang-format)
fi

if [ "$CLANG_FORMAT" = "" ]
then
echo -e "\e[31m\e[1mFAILED to find clang-format. \e[0m"
echo "Please run the KITcar Init script or install clang-format-3.6"
echo -e "commit aborted. \nif you REALLY want to commit, do \"git commit --no-verify\" (be careful)"
exit 1
fi

(
#temporarily change directory to the root directory of the git repository to avoid issues with relative/absolute paths (git ls-files uses relative paths, git diff paths relative to the repository root)
cd "$(git rev-parse --show-toplevel)" || exit
fail=0
any_file_working_tree_unformatted=0
files_unformatted_in_working_tree=()
echo "Checking formatting of files to be committed"
CPP_FILES=$(git diff --cached --name-only --diff-filter=ACM | grep -e '\.c"\?$' -e '\.cpp"\?$' -e '\.h"\?$' -e '\.hpp"\?$' -e '\.cxx"\?$')
#change internal field separator to newline to avoid issues with paths with spaces
OIFS="$IFS"
IFS=$'\n'
for file in $CPP_FILES
do
  IFS=$OIFS
  echo -n -e "File: \e[1m$file\e[21m"
  #find the hash of the blob representing the current index state of $file
  index_hash=$(git ls-files -s "$file" | awk '$3=="0" {number_of_hits++; print $2} END {if(number_of_hits!=1) {exit 1}}')
  no_unique_index_hash=$?
 if [ $no_unique_index_hash = 1 ]
  then
   echo "failed to find hash to index file"
   exit 1
  fi
  git show "$index_hash" | $CLANG_FORMAT -style=file -assume-filename="$file" | diff <(git show "$index_hash") - > /dev/null
  index_unformatted=$?
  $CLANG_FORMAT -style=file "$file" | diff "$file" - > /dev/null
  working_tree_unformatted=$?
  if [ $working_tree_unformatted = 1 ]
  then
    any_file_working_tree_unformatted=1
    files_unformatted_in_working_tree+=("$file")
  fi
  if [ $index_unformatted = 1 ]
  then
	echo -e " \e[31mFAIL\e[0m"
	fail=1
  else
	echo -e " \e[32mclean\e[0m"
  fi
done
IFS=$OIFS

if [ $fail = 1 ]
then
  if [ $any_file_working_tree_unformatted = 1 ]
  then
    echo -e "Some files are not correcty formatted. The commit was \e[31mABORTED\e[0m.  Please format the code according to the Google C++ Style Guide and commit it again."
    echo "--"
    echo "To format the Files automatically, execute the following statement:"
    echo -n "$CLANG_FORMAT -style=file -i "
    for file in "${files_unformatted_in_working_tree[@]}"; do echo -n "'$(git rev-parse --show-toplevel)/$file' "; done
    echo ""
  else
     echo -e "The working tree is correctly formatted, but the index is not. The commit was \e[31mABORTED\e[0m."
     echo "You probably forgot to add the files to the index after formatting the code. Use \"git status\" to see what has changed and then add the formatted files with \"git add <file>...\" to the index."
     echo ""
  fi
fi
exit $fail
)
