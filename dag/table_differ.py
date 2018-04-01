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


def main():
  MODES = {
    "tgroup_hcount": lambda results: normtime_all_algo(results, get_taskfile_group, get_host_count),
    "tgroup_hcount_exp": lambda results: etime_static_algo(results, get_taskfile_group, get_host_count),
    "bandwidth_hcount": lambda results: normtime_all_algo(results, get_host_bandwidth, get_host_count),
    "bandwidth_hcount_exp": lambda results: etime_static_algo(results, get_host_bandwidth, get_host_count)
  }

  parser = argparse.ArgumentParser(description="Experiment results analysis")
  parser.add_argument("input_file_1", type=str, help="experiment results")
  parser.add_argument("input_file_2", type=str, help="experiment results")
  parser.add_argument("-m", "--mode", type=str, default="tgroup_hcount", choices=list(MODES.keys()), help="processing mode")
  args = parser.parse_args()

  with open(args.input_file_1) as input_file_1:
    results1 = json.load(input_file_1)
  with open(args.input_file_2) as input_file_2:
    results2 = json.load(input_file_2)

  table1, algo1 = MODES.get(args.mode)(results1)
  table2, algo2 = MODES.get(args.mode)(results2)

  print_diff(table1, table2, algo1)

ALL_FIELD = "makespan"
ALL_REFERENCE_ALGO_IDX = 0
def normtime_all_algo(results, cond1, cond2):
  ALGO_ORDER = ["OLB", "MCT", "BatchMin", "BatchMax", "BatchSufferage", "HCPT", "HEFT", "Lookahead", "PEFT"]
  # evaluate normalized results
  for task, bytask in groupby(results, get_taskfile_name):
    for platform, byplat in groupby(bytask, get_platform_name):
      algorithm_results = groupby(byplat, get_algorithm, False)
      assert len(algorithm_results[ALGO_ORDER[ALL_REFERENCE_ALGO_IDX]]) == 1
      reference = algorithm_results[ALGO_ORDER[ALL_REFERENCE_ALGO_IDX]][0]
      for algorithm, byalg in algorithm_results.items():
        byalg[0]["result"] = byalg[0][ALL_FIELD] / reference[ALL_FIELD]

  return agregate_results(results, ALGO_ORDER, cond1, cond2)


def etime_static_algo(results, cond1, cond2):
  ALGO_ORDER = ["HCPT", "HEFT", "Lookahead", "PEFT"]
  REFERENCE_ALGO = "HEFT"

  results = list(filter(lambda r: get_algorithm(r) in ALGO_ORDER, results))
  # evaluate normalized results
  for task, bytask in groupby(results, get_taskfile_name):
    for platform, byplat in groupby(bytask, get_platform_name):
      algorithm_results = groupby(byplat, get_algorithm, False)
      assert len(algorithm_results[REFERENCE_ALGO]) == 1
      reference = algorithm_results[REFERENCE_ALGO][0]
      for algorithm, byalg in algorithm_results.items():
        byalg[0]["result"] = byalg[0]["expected_makespan"] / reference["expected_makespan"]

  return agregate_results(results, ALGO_ORDER, cond1, cond2)

def agregate_results(results, algorithms, cond1, cond2):
  final_result = {}
  for c1, bycond1 in sorted(groupby(results, cond1)):
    final_result[c1] = {}
    for c2, bycond2 in sorted(groupby(bycond1, cond2)):
      final_result[c1][c2] = {}
      for algorithm, byalg in sorted(groupby(bycond2, get_algorithm), key=lambda pair: algorithms.index(pair[0])):
        #print(c1, c2, algorithm)
        mean = numpy.mean([r["result"] for r in byalg])
        std = numpy.std([r["result"] for r in byalg])
        final_result[c1][c2][algorithm] = [mean, std]
  #print("\n\n\n\n\n\n")
  return final_result, algorithms

def print_diff(table1, table2, algorithms):
  # print results as latex table
  # a lot of hardcode there for now. not sure if can be avoided without excessively generic code.
  print(par(r"""
  \begin{table}
  \caption{TODO}
  \begin{center}
     \small\begin{tabular}{*{%d}{l}}
  \toprule
  GROUP 2 TODO & %s \\ \midrule
  """) % (len(algorithms) + 1, " &  ".join(algorithms),))
  for c1 in table1:
    print(par(r"""
    \multicolumn{%d}{l}{GROUP 1 TODO %s} \\ \midrule
    """) % (len(algorithms) + 1, c1))
    for c2 in table1[c1]:
      print("%-15s" % c2, end="")
      for algorithm in sorted(table1[c1][c2], key=lambda alg: algorithms.index(alg)):
        print(" & %6.4f (%6.4f)" % (table1[c1][c2][algorithm][0] - table2[c1][c2][algorithm][0],
                                    table1[c1][c2][algorithm][1] - table2[c1][c2][algorithm][1]), end="")
      print(r" \\")
    print(r"\midrule")
  print(par(r"""
  \bottomrule
  \end{tabular}
  \end{center}
  \label{tab:TODO}
  \end{table}
  """))

if __name__ == "__main__":
  main()
