#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

pushd "$SCRIPT_DIR/.."

if [ ! -f "$SCRIPT_DIR/exp1.json" ]; then
  DYLD_LIBRARY_PATH=$HOME/github/pysimgrid/opt/SimGrid/lib/  python3 -m pysimgrid.tools.experiment "$SCRIPT_DIR/plat_exp1" "$SCRIPT_DIR/tasks_exp1" "$SCRIPT_DIR/algorithms.json" dag/exp1.json -j8 --simgrid-log-level=error
fi

if [ ! -f "$SCRIPT_DIR/exp1_inf.json" ]; then
  DYLD_LIBRARY_PATH=$HOME/github/pysimgrid/opt/SimGrid/lib/  python3 -m pysimgrid.tools.experiment "$SCRIPT_DIR/plat_exp1_inf" "$SCRIPT_DIR/tasks_exp1" "$SCRIPT_DIR/algorithms.json" dag/exp1_inf.json -j8 --simgrid-log-level=error
fi

for (( idx=1; idx<=15; idx++ ))
do
    if [ ! -f "$SCRIPT_DIR/exp2_$idx.json" ]; then
      DYLD_LIBRARY_PATH=$HOME/github/pysimgrid/opt/SimGrid/lib/  python3 -m pysimgrid.tools.experiment "$SCRIPT_DIR/plat_exp2" "$SCRIPT_DIR/tasks_exp2_$idx" "$SCRIPT_DIR/algorithms.json" dag/exp2_$idx.json -j8
    fi

    if [ ! -f "$SCRIPT_DIR/exp2_inf_$idx.json" ]; then
      DYLD_LIBRARY_PATH=$HOME/github/pysimgrid/opt/SimGrid/lib/  python3 -m pysimgrid.tools.experiment "$SCRIPT_DIR/plat_exp2_inf" "$SCRIPT_DIR/tasks_exp2_$idx" "$SCRIPT_DIR/algorithms.json" dag/exp2_inf_$idx.json -j8
    fi
done

if [ ! -f "$SCRIPT_DIR/exp3.json" ]; then
  DYLD_LIBRARY_PATH=$HOME/github/pysimgrid/opt/SimGrid/lib/  python3 -m pysimgrid.tools.experiment "$SCRIPT_DIR/plat_exp3" "$SCRIPT_DIR/tasks_exp3" "$SCRIPT_DIR/algorithms.json" dag/exp3.json -j8
fi

popd
