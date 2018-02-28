#! /usr/bin/env bash
##########################################################################
# Copyright (c) 2017 German Aerospace Center (DLR). All rights reserved. #
# SPDX-License-Identifier: BSD-2-Clause                                  #
##########################################################################

# Helper script for running the experiment_coordinator.py script with a fixed PYTHONHASHSEED.

# Get the path to the actual folder this script is in, so the python script can be found
SOURCE="${BASH_SOURCE[0]}"
while [ -h "$SOURCE" ]; do # resolve $SOURCE until the file is no longer a symlink
  DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE" # if $SOURCE was a relative symlink, we need to resolve it relative to the path where the symlink file was located
done
DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"

# Set PYTHONHASHSEED to some fixed value and start the python script
PYTHONHASHSEED='42' python3 "$DIR/experiment_coordinator.py" $@
