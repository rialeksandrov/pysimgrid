#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import os

def run_experiment(plat, task, config_path, folder):
  plat_name = plat["name"]
  plat_path = plat["path"]
  task_name = task["name"]
  task_path = task["path"]

  plat_path = os.path.realpath(os.path.join(os.path.dirname(__file__), plat_path))
  task_path = os.path.realpath(os.path.join(os.path.dirname(__file__), task_path))
  result_json = os.path.realpath(os.path.join(os.path.dirname(__file__), folder, plat_name + "_" + task_name + ".json"))

  # print(plat_path, task_path, result_path)

  if os.path.exists(result_json):
    print(result_json + " already exist")
    return
  if os.path.exists(plat_path) == False:
    print(plat_path + " doesn't exist")
    return
  if os.path.exists(task_path) == False:
    print(task_path + " doesn't exist")
    return

  os.system("cd .. && DYLD_LIBRARY_PATH=$HOME/github/pysimgrid/opt/SimGrid/lib/ python3 -m pysimgrid.tools.experiment "
            + plat_path + " " + task_path + " " + config_path + " " + result_json + " -j8")




def main():
  parser = argparse.ArgumentParser(description="Run experiments")
  parser.add_argument("config", type=str, help="path to json defining the experiment")
  args = parser.parse_args()
  if os.path.exists(args.config) == False:
    print("Error - config file doesn't exist")
    return

  with open(args.config, "r") as file:
    config_path = os.path.realpath(args.config)
    # print(config_path)
    config = json.load(file)

    experiment_name = config["simulation"]["name"]
    data_path = os.path.realpath(os.path.join(os.path.dirname(__file__), experiment_name))
    if os.path.exists(data_path) == False:
      os.mkdir(data_path)

    tasks = config["tasks"]
    if not isinstance(tasks, list):
      tasks = [tasks]

    platforms = config["platforms"]
    if not isinstance(platforms, list):
      platforms = [platforms]
    for plat in platforms:
      for task in tasks:
        run_experiment(plat, task, config_path, experiment_name)





if __name__ == "__main__":
  main()

