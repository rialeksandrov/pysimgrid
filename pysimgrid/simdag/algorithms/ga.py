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

from .batch import BatchMin, BatchMax, BatchSufferage
from .dls import DLS
from .hcpt import HCPT
from .heft import HEFT
from .lookahead import Lookahead
from .mct import MCT
from .olb import OLB
from .peft import PEFT
from .random import RandomStatic
from .round_robin import RoundRobinStatic
from .simheft import SimHEFT

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
    self.score = 0

    self.matching = {}
    for task in self.tasks:
      self.matching[task] = random.choice(self.hosts)

    self.scheduling = {}
    for i in range(len(self.tasks)):
      self.scheduling[self.tasks[i]] = i

  def copy_schedule(self, schedule_and_span):
    self.matching = {}
    self.scheduling = {}
    id = 0
    for host in self.hosts:
      for task in schedule_and_span[0][host]:
        self.matching[task] = host
        self.scheduling[task] = id
        id = id + 1

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
    print(cut_off)
    for task in self.tasks:
      if cut_off == 0:
        gen.matching[task], self.matching[task] = self.matching[task], gen.matching[task]
      else:
        cut_off -= 1
    #print (gen.matching)

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

  def get_schedule_by_name(self):
    schedule = {host.name: [] for host in self.hosts}
    order = self.make_task_order()
    for i in range(len(order)):
      task = order[i]
      schedule[self.matching[task].name].append(task.name)
      # print("host :", self.matching[task].name, " task: ", task.name)
    return schedule

  def get_schedule(self):
    schedule = {host: [] for host in self.hosts}
    order = self.make_task_order()
    for i in range(len(order)):
      task = order[i]
      schedule[self.matching[task]].append(task)
      # print("host :", self.matching[task].name, " task: ", task.name)
    return schedule

  def __eq__(self, other):
    return self.scheduling == other.scheduling and self.matching == other.matching





def _restore_schedule(simulation, serialized):
  tasks = {t.name: t for t in simulation.tasks}
  hosts = {h.name: h for h in simulation.hosts}
  result = {}
  end_scheduled = False
  for hostname, tasknames in serialized.items():
    tasklist = []
    for taskname in tasknames:
      end_scheduled = end_scheduled or taskname == "end"
      tasklist.append(tasks[taskname])
    result[hosts[hostname]] = tasklist
  if not end_scheduled:
    for host_schedule in result.values():
      host_schedule.append(tasks["end"])
      break
  return result, end_scheduled

def _evaluation(platform_path, tasks_path, schedule_by_name):
  import logging
  from .. import simulation
  logging.getLogger().setLevel(logging.WARNING)
  with simulation.Simulation(platform_path, tasks_path, log_config="root.threshold:WARNING") as simulation:
    restored_schedule_state, final = _restore_schedule(simulation, schedule_by_name)
    scheduler = _ExtrenalSchedule(simulation, restored_schedule_state)
    scheduler.run()
    finish_time = 0
    for t in simulation.tasks:
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

def _update_subgraph(full, subgraph, task):
  parents = full.pred[task]
  subgraph.add_node(task)
  for parent, edge_dict in parents.items():
    subgraph.add_edge(parent, task, edge_dict)


def _return_non_random_schedules(simulation):
  res = []
  classes = [DLS(simulation), HCPT(simulation), HEFT(simulation),
             Lookahead(simulation), PEFT(simulation), RoundRobinStatic(simulation)]
  for cl in classes:
    res.append(cl.get_schedule(simulation))

  return res


def _weighted_choice(sum_weights, weights):
  r = random.uniform(0, sum_weights)
  upto = 0
  for idx in range(len(weights)):
    w = weights[idx]
    if upto + w >= r:
      return idx
    upto += w
  assert False, "Shouldn't get here"

class GA(StaticScheduler):
  """
  Genetic-Algorithm-Based Approach

  """


  def get_schedule(self, simulation):
    """
    Overridden.
    """
    random.seed(12345)
    POPULATION_SIZE = 10
    NUMBER_OF_ITERATION = 2
    RANK_CONSTANT = 1.1
    CROSSOVER_PROBABILITY = 1.0



    nxgraph = simulation.get_task_graph()
    platform_model = cscheduling.PlatformModel(simulation)
    state = cscheduling.SchedulerState(simulation)

    topological_order = networkx.topological_sort(nxgraph, reverse=False)

    ctx = multiprocessing.get_context("spawn")

    subgraph = networkx.DiGraph()
    for task in nxgraph:
      _update_subgraph(nxgraph, subgraph, task)

    schedules = _return_non_random_schedules(simulation)

    popultaion = []
    for schedule in schedules:
      gen = Chromosome(topological_order, simulation.hosts, nxgraph)
      gen.copy_schedule(schedule)
      popultaion.append(gen)

    for i in range(POPULATION_SIZE - len(popultaion)):
      gen = Chromosome(topological_order, simulation.hosts, nxgraph)
      popultaion.append(gen)

    weights = []
    for i in range(POPULATION_SIZE):
      weights.append(RANK_CONSTANT ** (POPULATION_SIZE - i - 1) * (RANK_CONSTANT - 1))

    sum_weights = sum(weights)

    for epoh in range(NUMBER_OF_ITERATION):
      print ("Starting #", epoh, ":")
      with tempfile.NamedTemporaryFile("w", suffix=".dot") as graph_file:
        _serialize_graph(subgraph, graph_file)
        for gen in popultaion:
          if epoh == 1:
            print (gen.get_schedule_by_name())
            print(gen.scheduling)
            print(gen.make_task_order())
          with ctx.Pool(1) as process:
            ans = process.apply(_evaluation, (simulation.platform_path, graph_file.name, gen.get_schedule_by_name()))
          gen.score = ans
          print("ans :", ans)


      # Selection
      popultaion = sorted(popultaion, key = lambda t: t.score)
      new_population = []
      for idx in range(POPULATION_SIZE):
        new_population.append(copy.copy(popultaion[_weighted_choice(sum_weights, weights)]))
      popultaion = new_population

      #Crossover
      new_population = []
      for idx in range(POPULATION_SIZE // 2):
        gen1 = random.choice(popultaion)
        popultaion.remove(gen1)
        gen2 = random.choice(popultaion)
        popultaion.remove(gen2)
        prob = random.uniform(0, 1)
        if prob < CROSSOVER_PROBABILITY:
          gen1.matching_crossover(gen2)
          # gen1.scheduling_crossover(gen2)
        new_population.append(copy.copy(gen1))
        new_population.append(copy.copy(gen2))
      print (len(popultaion))
      popultaion = new_population




    return popultaion[0].get_schedule()

