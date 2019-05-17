#!/bin/bash
n=1
while ! mkdir archive/run_$n >/dev/null 2>&1
do
  n=$((n+1))
done
var=`ls archive | sort -nr | head -n 1`
cp -r logs/* archive/${var}/
