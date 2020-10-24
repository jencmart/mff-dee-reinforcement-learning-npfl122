#!/bin/bash
TO="$PWD/tasks"
FROM="ufal"

if [ "$#" -ne 1 ]; then
    echo "Illegal number of arguments. Correct example: './update.sh 04' "
    exit 1
fi

# 0. update github + check that passed argument is actually existing lab
echo "...pulling from github...."
cd $FROM
git pull
if [[ -d "labs/$1" ]] ; then
    echo "Task $1 found!"
else
    echo "Task $1 not found. Exiting now."
    exit 1    
fi

# And also check that target not already exists
if [[ -d "$TO/task$1" ]] ; then
    echo "Warn: pycharm dir allready contains task$1!"
    while true; do
        read -p "Do you want to continue[Y/n]?" yn
        case $yn in
            [Yy]* ) break;;
            [Nn]* ) exit;;
            * ) echo "Please answer yes or no.";;
        esac
    done
fi

# 1. Create directory for new Task
if mkdir -p "$TO/task$1" ; then
    echo "1/3 Created directory for task $1"
else
    echo "Error while creating driectory $1"
    exit 1
fi

# 2. Move tasks
cd tasks
if mv ./*.md $TO/task$1 2>/dev/null ; then
    echo "2/3 Successfuly moved task descriptions"
else
    echo "2/3 No task descriptions to move..."
fi

# 3. Move python source code
cd ..
cd labs
cd $1
if mv ./*.py $TO/task$1 2>/dev/null ; then
    echo "3/3 Successfuly moved source codes"
else
    echo "3/3 No source codes to move..."
fi

