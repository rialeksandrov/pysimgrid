# This file is part of pysimgrid, a Python interface to the SimGrid library.
#
# Copyright 2015-2016 Alexey Nazarenko and contributors
#
# This library is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# along with this library.  If not, see <http://www.gnu.org/licenses/>.
#

"""
Simple, but effective batch simulation tool.

Usage::

    python -m pysimgrid.tools.experiment [-h] [-j JOBS] [-l {debug,info,warning,error,critical}]
                         [--simgrid-log-level {trace,debug,verbose,info,warning,error,critical}]
                         [--stop-on-error] [--algo [ALGO [ALGO ...]]]
                         platforms tasks algorithms output

    positional arguments:
      platforms             path to file or directory containing platform
                            definitions (*.xml)
      tasks                 path to file or directory containing task definitions
                            (*.dax, *.dot)
      algorithms            path to json defining the algorithms to use
      output                path to the output file

    optional arguments:
      -h, --help            show this help message and exit
      -j JOBS, --jobs JOBS  number of parallel jobs to run
      -l {debug,info,warning,error,critical}, --log-level {debug,info,warning,error,critical}
                            job log level
      --simgrid-log-level {trace,debug,verbose,info,warning,error,critical}
                            simulator log level
      --stop-on-error       stop experiment on a first error
      --algo [ALGO [ALGO ...]]
                            name(s) of algorithms to use (case-sensitive; must
                            still be defined in algorithms file)
"""

from __future__ import print_function

import argparse
import collections
import datetime
import fnmatch
import itertools
import json
import logging
import multiprocessing
import os
import textwrap
import time
import traceback
import ntpath

from .. import simdag


# Weird, but necessary workaround to use nested multiprocessing.
#
# Usecase: 
# * experiment tool uses process per experiment
# * some algorithms use processes to run SimGrid to assess partial schedules
#

import multiprocessing.pool


class NoDaemonProcess(multiprocessing.context.SpawnProcess):
    @property
    def daemon(self):
        return True

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

######################


def file_list(file_or_dir, masks=["*"]):
  if not os.path.exists(file_or_dir):
    raise Exception("path %s does not exist" % file_or_dir)
  if os.path.isdir(file_or_dir):
    result = []
    listing = os.listdir(file_or_dir)
    for fname in listing:
      if any((fnmatch.fnmatch(fname, pattern) for pattern in masks)):
        result.append(os.path.join(file_or_dir, fname))
    return result
  else:
    return [os.path.abspath(file_or_dir)]


def import_algorithm(algorithm):
  name_parts = algorithm.split(".")
  module_name = ".".join(name_parts[:-1])
  module = __import__(module_name)
  result = module
  for name in name_parts[1:]:
    result = getattr(result, name)
  assert isinstance(result, type)
  return result

def make_gantt(simultation, platform, tasks, algorithm_class, args):
  if args["print_plotly_gantt"] == False and args["print_matplotlib_gantt"] == False:
    return
  _, platform_name = ntpath.split(platform)
  _, task_name = ntpath.split(tasks)
  name = str(platform_name) + "_" + str(task_name) + "_" + str(algorithm_class)
  if args["print_plotly_gantt"]:
    import plotly.plotly as py
    import plotly.figure_factory as ff
    colors = {'Not Started': 'rgb(220, 0, 0)',
              'Incomplete': (1, 0.9, 0.16),
              'Complete': 'rgb(0, 255, 100)'}
    df = []
    for task in simultation.tasks:
      host = task.hosts
      assert (len(host) == 1)
      to_add = dict(Task=str(host[0].name),
                    Start=datetime.datetime.fromtimestamp(int(task.start_time)).strftime('%Y-%m-%d %H:%M:%S'),
                    Finish=datetime.datetime.fromtimestamp(int(task.finish_time)).strftime('%Y-%m-%d %H:%M:%S'),
                    Resource='Complete')
      df.append(to_add)

    py.sign_in(args["plotly_login"]["username"], args["plotly_login"]["token"])

    fig = ff.create_gantt(df, colors=colors, index_col='Resource', show_colorbar=True, group_tasks=True)
    py.image.save_as(fig, name + "_plotly.jpeg")
  if args["print_matplotlib_gantt"]:
    import pandas as pd
    import matplotlib.pyplot as plt
    colors = {'Not Started': "gold",
              'Incomplete': "red",
              'Complete': "limegreen"}
    df = []
    for task in simultation.tasks:
      host = task.hosts
      assert (len(host) == 1)
      to_add = dict(Task=str(host[0].name),
                    Start=task.start_time,
                    Finish=task.finish_time,
                    Resource='Complete')
      df.append(to_add)
    df = pd.DataFrame(df)
    df["Diff"] = df.Finish - df.Start
    fig, ax = plt.subplots(figsize=(6, 3))

    labels = []
    for i, task in enumerate(df.groupby("Task")):
      labels.append(task[0])
      for r in task[1].groupby("Resource"):
        data = r[1][["Start", "Diff"]]
        ax.broken_barh(data.values, (i - 0.4, 0.8), color=colors[r[0]])

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("time [ms]")
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig(name + "_matplotlib.png", dpi=400)

def run_experiment(job):
  platform, tasks, algorithm, config, args = job
  python_log_level, simgrid_log_level = config["log_level"], config["simgrid_log_level"]
  stop_on_error = config["stop_on_error"]
  logging.getLogger().setLevel(python_log_level)
  logger = logging.getLogger("pysimgrid.tools.Experiment")
  logger.info("Starting experiment (platform=%s, tasks=%s, algorithm=%s)", platform, tasks, algorithm["class"])
  scheduler_class = import_algorithm(algorithm["class"])
  # init return values with NaN's
  makespan, exec_time, comm_time, sched_time, exp_makespan = [float("NaN")] * 5
  try:
    with simdag.Simulation(platform, tasks, log_config="root.threshold:" + simgrid_log_level, args=args) as simulation:
      scheduler = scheduler_class(simulation)
      scheduler.run()
      makespan = simulation.clock
      exec_time = sum([t.finish_time - t.start_time for t in simulation.tasks])
      comm_time = sum([t.finish_time - t.start_time for t in simulation.all_tasks[simdag.TaskKind.TASK_KIND_COMM_E2E]])
      sched_time = scheduler.scheduler_time
      make_gantt(simulation, platform, tasks, algorithm["class"], args)
      if scheduler.expected_makespan is not None:
        exp_makespan = scheduler.expected_makespan
  except Exception:
    # output is not pretty, but complete and robust. it is a crash anyway.
    #   note the wrapping of job into a tuple
    message = "Simulation failed! Parameters: %s" % (job,)
    if stop_on_error:
      raise Exception(message)
    else:
      logger.exception(message)
  return job, makespan, exec_time, comm_time, sched_time, exp_makespan


def progress_reporter(iterable, length, logger):
  start_time = last_result_timestamp = time.time()
  average_time = 0.
  for idx, element in enumerate(iterable):
    current = time.time()
    elapsed = current - last_result_timestamp
    last_result_timestamp = current
    count = idx + 1
    average_time = average_time * (count - 1) / float(count) + elapsed / count
    remaining = (length - idx) * average_time
    eta_string = (" [ETA: %s]" % datetime.timedelta(seconds=remaining)) if idx > 10 else ""
    logger.info("%d/%d%s", idx + 1, length, eta_string)
    yield element
  logger.info("Finished. %d experiments in %f seconds", length, time.time() - start_time)


def main():
  _LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
  _LOG_FORMAT = "[%(name)s] [%(levelname)5s] [%(asctime)s] %(message)s"
  _LOG_LEVEL_FROM_STRING = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL
  }

  parser = argparse.ArgumentParser(description="Run experiments for a set of scheduling algorithms")
  parser.add_argument("platforms", type=str, help="path to file or directory containing platform definitions (*.xml)")
  parser.add_argument("tasks", type=str, help="path to file or directory containing task definitions (*.dax, *.dot)")
  parser.add_argument("config", type=str, help="path to json defining the experiment config")
  parser.add_argument("output", type=str, help="path to the output file")
  parser.add_argument("-j", "--jobs", type=int, default=1, help="number of parallel jobs to run")
  parser.add_argument("-l", "--log-level", type=str, choices=["debug", "info", "warning", "error", "critical"],
                      default="warning", help="job log level")
  parser.add_argument("--simgrid-log-level", type=str, choices=["trace", "debug", "verbose", "info", "warning", "error", "critical"],
                      default="warning", help="simulator log level")
  parser.add_argument("--stop-on-error", action="store_true", default=False, help="stop experiment on a first error")
  # useful for algorithm debugging
  parser.add_argument("--algo", type=str, nargs="*", help="name(s) of algorithms to use (case-sensitive; must still be defined in algorithms file)")
  args = parser.parse_args()

  logging.basicConfig(level=logging.DEBUG, format=_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
  logger = logging.getLogger("Experiment")

  with open(args.config, "r") as file:
    config = json.load(file)
    algorithms = config["algorithms"]
    if not isinstance(algorithms, list):
      algorithms = [algorithms]
    simulation_args = config["simulation"]
    if not isinstance(simulation_args, list):
      simulation_args = [simulation_args]
    # we can allow non-unique algorithm names. but really shoudn't as it can mess up the results.
    algorithm_names = set([a["name"] for a in algorithms])
    if len(algorithms) != len(algorithm_names):
      raise Exception("algorithm names should be unique")
    if args.algo:
      for algo_name in args.algo:
        if algo_name not in algorithm_names:
          raise Exception("algorithm %s is not defined" % algo_name)
      algorithms = [a for a in algorithms if a["name"] in args.algo]

  platforms = file_list(args.platforms, ["*.xml"])
  tasks = file_list(args.tasks)
  config = [{
    "stop_on_error": args.stop_on_error,
    "log_level": _LOG_LEVEL_FROM_STRING[args.log_level],
    "simgrid_log_level": args.simgrid_log_level
  }]

  # convert to list just get length nicely
  #   can be left as an iterator, but memory should not be the issue
  jobs = list(itertools.product(platforms, tasks, algorithms, config, simulation_args))

  # report experiment setup
  #   looks scary, but it's probably a shortest way to do this in terms of LOC
  logger.info(textwrap.dedent("""\
  Starting the experiment.
    Total runs: %d

    Platform source: %s
    Platform count:  %d

    Tasks source:    %s
    Tasks count:     %d

    Algorithms count: %d
    Algorithms:
  %s

    Configuration:
  %s
  """) % (len(jobs), args.platforms, len(platforms), args.tasks, len(tasks), len(algorithms),
          "\n".join(["    " + a["name"] for a in algorithms]),
          "\n".join(["    %s: %s" % (k, v) for k,v in config[0].items()])
  ))

  results = []
  # using the spawn context is important
  #    by default, multiprocessing uses fork, which conflicts with coolhacks inside SimGrid/XBT (library constructors)
  # in more details:
  #    SimGrid library init stores the id of the main thread & creates some custom TLS (don't ask me why)
  #    'fork' doesn't lead to the reinit
  #    SimGrid crashes on unitialized TLS
  ctx = multiprocessing.get_context("spawn")
  with NoDaemonPool(processes=args.jobs, maxtasksperchild=1, context=ctx) as pool:
    for job, makespan, exec_time, comm_time, sched_time, exp_makespan in progress_reporter(pool.imap_unordered(run_experiment, jobs, 1), len(jobs), logger):
      platform, tasks, algorithm, _, _ = job
      results.append({
        "platform": platform,
        "tasks": tasks,
        "algorithm": algorithm,
        "makespan": makespan,
        "exec_time": exec_time,
        "comm_time": comm_time,
        "sched_time": sched_time,
        "expected_makespan": exp_makespan
      })


  with open(args.output, "w") as out_file:
    json.dump(results, out_file)


if __name__ == "__main__":
  main()
