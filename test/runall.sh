#!/bin/bash

# Run the unittests with 1 up to specified number of processors
#
# If no processors or invalid number specified, defaults to one
#
# Usage: 'runall.sh number'


# Default number of processors to use if incorrect/invalid number
# specified
defaultCount=1;

if [ $# -eq 0 ]
then
    # if no argument is passed, use default value
    procCount=$defaultCount;
else
    procCount=$1;
fi

# validate proc count request
re='^[0-9]+$'
if ! [[ $procCount =~ $re ]] ; then
    echo "Invalid proc count. Defaulting to $defaultCount."
    procCount=$defaultCount;
fi

# begin tests
for n in `seq 1 $procCount`; do
    printf "\n[Running with $n processors]\n\n";
    args="-n $n ./petscpp-tests";
    mpiexec $args;
done
