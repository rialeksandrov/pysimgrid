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

from collections import deque

import networkx
import numpy
import copy
import random
import operator
import itertools
import multiprocessing
import tempfile

from ... import cscheduling
from ..scheduler import StaticScheduler


class _ExtrenalSchedule(StaticScheduler):
  def __init__(self, simulation, schedule):
    super(_ExtrenalSchedule, self).__init__(simulation)
    self.__schedule = schedule

    def get_schedule(self, simulation):
        return self.__schedule

class Chromosome:

  def __init__(self, topological_order, hosts, nxgraph):
    self.nxgraph = nxgraph
    self.tasks = topological_order
    self.hosts = hosts
    self.matching = {}

    for task in self.tasks:
      self.matching[task] = random.choice(self.hosts)
      self.scheduling = {}

    for i in range(len(self.tasks)):
      self.scheduling[self.tasks[i]] = i

    def scheduling_mutation(self):
      task = random.choice(self.tasks)
      left_boundary = 0
      right_boundary = len(self.tasks) - 1
      for child, edge in self.nxgraph[task].items():
        right_boundary = min(right_boundary, self.scheduling[child] - 1)

      for parent in self.nxgraph.pred[task]:
        left_boundary = max(left_boundary, self.scheduling[parent] + 1)

      new_id = random.randint(left_boundary, right_boundary)
      old_id = self.scheduling[task]

      add = 1
      if new_id > old_id:
        add = -1

      for item in self.tasks:
        if min(old_id, new_id) <= self.scheduling[item] and self.scheduling[item] <= max(old_id, new_id):
          self.scheduling[item] += add

      self.scheduling[task] = new_id

  def matching_mutation(self):
    task = random.choice(self.tasks)
    self.matching[task] = random.choice(self.hosts)

  def make_task_order(self):
    res = copy.copy(self.tasks)
    for task in self.scheduling:
      res[self.scheduling[task]] = task
    return res

  def matching_crossover(self, gen):
    cut_off = random.randint(0, len(self.tasks) - 1)
    order = self.make_task_order()
    for task in order:
      if cut_off == 0:
        temp = gen.matching[task]
        gen.matching[task] = self.matching[task]
        self.matching[task] = temp
      else:
        cut_off -= 1

  def scheduling_crossover(self, gen):
    cut_off = random.randint(0, len(self.tasks) - 1)
    self_order = self.make_task_order()
    gen_order = gen.make_task_order()
    first_res = {}
    for i in range(cut_off):
      task = gen_order[i]
      first_res[task] = i

    id = cut_off
    for task in self_order:
      if task not in first_res:
        first_res[task] = id
        id += 1

    second_res = {}
    for i in range(cut_off):
      task = self_order[i]
      second_res[task] = i

    id = cut_off
    for task in gen_order:
      if task not in second_res:
        second_res[task] = id
        id += 1

    gen.scheduling = first_res
    self.scheduling = second_res

  def get_schedule(self):
    schedule = {host: [] for host in self.hosts}
    order = self.make_task_order()
    for task in order:
      schedule[self.matching[task]].append(task)

  def __eq__(self, other):
    return self.scheduling == other.scheduling and self.matching == other.matching


def _evaluation(platform_path, tasks_path, gen):
  import logging
  from .. import simulation
  schedule = gen.get_schedule()
  logging.getLogger().setLevel(logging.WARNING)
  with simulation.Simulation(platform_path, tasks_path, log_config="root.threshold:WARNING") as simulation:
    scheduler = _ExtrenalSchedule(simulation, schedule)
    scheduler.run()
    finish_time = 0
    for t in simulation:
      finish_time = max(finish_time, t.finish_time)
    return finish_time


def _serialize_graph(graph, output_file):
  output_file.write("digraph G {\n")
  for task in graph:
    output_file.write('  "%s" [size="%f"];\n' % (task.name, task.amount))
  output_file.write("\n")
  for src, dst, data in graph.edges_iter(data=True):
    output_file.write('  "%s" -> "%s" [size="%f"];\n' % (src.name, dst.name, data["weight"]))
  output_file.write("}\n")
  output_file.flush()

class GA(StaticScheduler):
  """
  Genetic-Algorithm-Based Approach

  """


  def get_schedule(self, simulation):
    """
    Overridden.
    """

    nxgraph = simulation.get_task_graph()
    platform_model = cscheduling.PlatformModel(simulation)
    state = cscheduling.SchedulerState(simulation)

    ctx = multiprocessing.get_context("spawn")

    topological_order = networkx.topological_sort(nxgraph, reverse=True)
    graph_file = tempfile.NamedTemporaryFile("w", suffix=".dot")
    _serialize_graph(nxgraph, graph_file)

    gen = Chromosome(topological_order, simulation.hosts, nxgraph)
    with ctx.Pool(1) as process:
      ans = process.apply(_evaluation, (simulation.platform_path, graph_file.name, gen))
    print("ans, :", ans)
    return gen.get_schedule()

