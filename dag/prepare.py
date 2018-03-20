#!/usr/bin/env python3
from __future__ import print_function

import argparse
import json
import os


def gen_plat(config):
  name = config["name"]
  path = config["path"]
  parameters = config["parameters"]
  gen = parameters["generate"]
  if gen == False:
    # print(gen)
    return
  cur_dir = os.path.dirname(__file__)
  gen_path = os.path.realpath(os.path.join(cur_dir, path))
  if os.path.exists(gen_path):
    # print(gen_path)
    return
  os.system("cd .. && DYLD_LIBRARY_PATH=$HOME/github/pysimgrid/opt/SimGrid/lib/ python3 -m pysimgrid.tools.plat_gen " +
            gen_path + " " + parameters["args"])

def gen_tasks(config):
  name = config["name"]
  path = config["path"]
  parameters = config["parameters"]
  gen = parameters["generate"]
  if gen == False:
    # print(gen)
    return
  cur_dir = os.path.dirname(__file__)
  gen_path = os.path.realpath(os.path.join(cur_dir, path))
  if os.path.exists(gen_path):
    # print(gen_path)
    return
  os.system("cd .. && DYLD_LIBRARY_PATH=$HOME/github/pysimgrid/opt/SimGrid/lib/ python3 -m pysimgrid.tools.dag_gen " +
            gen_path + " " + parameters["args"])

  # print(name, path)


def main():
  parser = argparse.ArgumentParser(description="Prepare platforms and tasks")
  parser.add_argument("config", type=str, help="path to json defining the experiment")
  args = parser.parse_args()
  if os.path.exists(args.config) == False:
    print("Error - config file doesn't exist")
    return

  with open(args.config, "r") as file:
    config = json.load(file)

    tasks = config["tasks"]
    if not isinstance(tasks, list):
      tasks = [tasks]
    for task in tasks:
      gen_tasks(task)

    platforms = config["platforms"]
    if not isinstance(platforms, list):
      platforms = [platforms]
    for plat in platforms:
      gen_plat(plat)


if __name__ == "__main__":
  main()

