#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {1..100}
  do 
     python generate_random.py $i
 done