"""
Experiment results analysis tool.

Not yet generic. Should be included in pysimgrid.tools when is.
"""

from __future__ import print_function

import argparse
import collections
import json
import os
import textwrap

import numpy

import plotly.plotly as py
import plotly.graph_objs as go


def groupby(results, condition, asitems=True):
  groups = collections.defaultdict(list)
  for data in results:
    key = condition(data)
    groups[key].append(data)
  return groups.items() if asitems else groups


def par(string):
  return textwrap.dedent(string).strip()


def get_taskfile_name(item):
  return os.path.basename(item["tasks"]).rsplit(".", 1)[0]


def get_task_count(item):
  return int(get_taskfile_name(item).split("_")[1])


def get_taskfile_group(item):
  return "_".join(get_taskfile_name(item).split("_")[:2])


def get_platform_name(item):
  return os.path.basename(item["platform"])


def get_host_count(item):
  return int(get_platform_name(item).split("_")[1])


def get_host_bandwidth(item):
  return int(get_platform_name(item).split("_")[3])


def get_algorithm(item):
  return item["algorithm"]["name"]

ALGO_ORDER = []

def main():
  MODES = {
    "tgroup_hcount": lambda results: normtime_all_algo(results, get_taskfile_group, get_host_count),
    "tgroup_hcount_exp": lambda results: etime_static_algo(results, get_taskfile_group, get_host_count),
    "bandwidth_hcount": lambda results: normtime_all_algo(results, get_host_bandwidth, get_host_count),
    "bandwidth_hcount_exp": lambda results: etime_static_algo(results, get_host_bandwidth, get_host_count)
  }

  parser = argparse.ArgumentParser(description="Experiment results analysis")
  parser.add_argument("input_file", type=str, help="experiment results")
  parser.add_argument("config", type=str, help="path to json defining the experiment")
  parser.add_argument("-m", "--mode", type=str, default="tgroup_hcount", choices=list(MODES.keys()), help="processing mode")
  args = parser.parse_args()

  with open(args.config, "r") as file:
    config = json.load(file)
    for algo in config["algorithms"]:
      ALGO_ORDER.append(algo["name"])
    with open(args.input_file) as input_file:
      results = json.load(input_file)

    table = MODES.get(args.mode)(results)
    username = config["simulation"]["plotly_login"]["username"]
    token = config["simulation"]["plotly_login"]["token"]
    experiment_name = config["simulation"]["name"]
    data_path = os.path.realpath(os.path.join(os.path.dirname(__file__), experiment_name))
    if os.path.exists(data_path) == False:
      os.mkdir(data_path)
    print_result(table, username, token, data_path)

ALL_FIELD = "makespan"
def normtime_all_algo(results, cond1, cond2):
  ALL_REFERENCE_ALGO_IDX = 0
  for task, bytask in groupby(results, get_taskfile_name):
    for platform, byplat in groupby(bytask, get_platform_name):
      algorithm_results = groupby(byplat, get_algorithm, False)
      assert len(algorithm_results[ALGO_ORDER[ALL_REFERENCE_ALGO_IDX]]) == 1
      reference = algorithm_results[ALGO_ORDER[ALL_REFERENCE_ALGO_IDX]][0]
      for algorithm, byalg in algorithm_results.items():
        byalg[0]["result"] = byalg[0][ALL_FIELD]

  return agregate_results(results, cond1, cond2)


def etime_static_algo(results, cond1, cond2):
  REFERENCE_ALGO = ALGO_ORDER[0]

  results = list(filter(lambda r: get_algorithm(r) in ALGO_ORDER, results))
  # evaluate normalized results
  for task, bytask in groupby(results, get_taskfile_name):
    for platform, byplat in groupby(bytask, get_platform_name):
      algorithm_results = groupby(byplat, get_algorithm, False)
      assert len(algorithm_results[REFERENCE_ALGO]) == 1
      reference = algorithm_results[REFERENCE_ALGO][0]
      for algorithm, byalg in algorithm_results.items():
        byalg[0]["result"] = byalg[0]["expected_makespan"]

  return agregate_results(results, cond1, cond2)

def agregate_results(results, cond1, cond2):
  final_result = {}
  for c1, bycond1 in sorted(groupby(results, cond1)):
    final_result[c1] = {}
    for c2, bycond2 in sorted(groupby(bycond1, cond2)):
      final_result[c1][c2] = {}
      for algorithm, byalg in sorted(groupby(bycond2, get_algorithm), key=lambda pair: ALGO_ORDER.index(pair[0])):
        final_result[c1][c2][algorithm] = [r["result"] for r in byalg]
  return final_result

def print_result(table, username, token, folder):
  colors = ['hsl(' + str(h) + ',50%' + ',50%)' for h in numpy.linspace(0, 360, len(ALGO_ORDER))]
  py.sign_in(username, token)
  for c1 in table:
    for c2 in table[c1]:
      data = []
      idx = 0
      for algorithm in sorted(table[c1][c2], key=lambda alg: ALGO_ORDER.index(alg)):
        points = {
          'x': table[c1][c2][algorithm],
          'type': 'box',
          'boxpoints' : 'all',
          'marker': {'color': colors[idx]},
          'name':algorithm,
          'boxmean' : True
        }
        idx += 1
        data.append(points)
      layout = {'yaxis': {'showgrid': False, 'zeroline': False, 'tickangle': 60, 'showticklabels': False},
                'xaxis': {'zeroline': False, 'gridcolor': 'white'},
                'paper_bgcolor': 'rgb(233,233,233)',
                'plot_bgcolor': 'rgb(233,233,233)'
                }
      fig = go.Figure(data=data, layout=layout)
      py.image.save_as(fig, filename=folder + "/" +str(c1) + "_" + str(c2) + ".jpeg")


if __name__ == "__main__":
  main()
