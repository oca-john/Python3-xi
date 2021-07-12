#!/bin/sh
echo "Welcome to bash with python!"

python <<_EOF_PYTHON_SCRIPT
print("This is print by perl in bash")
_EOF_PYTHON_SCRIPT

echo "Back to bash now!"
