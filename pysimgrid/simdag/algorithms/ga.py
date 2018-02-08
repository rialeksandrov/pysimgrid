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
    self.score = -1

    self.matching = {}
    for task in self.tasks:
      self.matching[task] = random.choice(self.hosts)

    self.scheduling = {}
    for i in range(len(self.tasks)):
      self.scheduling[self.tasks[i]] = i

  def copy_schedule(self, schedule_and_span):
    self.matching = {}
    self.scheduling = {}

    for host in self.hosts:
      for task in schedule_and_span[0][host]:
        if task.name != "root" or task.name != "end":
          self.matching[task] = host

    indexes = {host : 0 for host in self.hosts}
    undone_parents = {task: len(self.nxgraph.pred[task]) for task in self.tasks}
    for task in self.tasks:
      for parent in self.nxgraph.pred[task]:
        if parent.name == "root":
          undone_parents[task] -= 1
          break

    for i in range(len(self.tasks)):
      task = self.tasks[0]
      ok = 0
      for host in self.hosts:
        if indexes[host] == len(schedule_and_span[0][host]):
          continue
        if undone_parents[schedule_and_span[0][host][indexes[host]]] == 0:
          task = schedule_and_span[0][host][indexes[host]]
          indexes[host] += 1
          ok = 1
          break
      assert ok == 1, "Did not find task to schedule!"
      self.scheduling[task] = i
      for child, edge in self.nxgraph[task].items():
        if child.name != "end":
          undone_parents[child] -= 1


  def scheduling_mutation(self):
    self.score = -1
    task = random.choice(self.tasks)
    left_boundary = 0
    right_boundary = len(self.tasks) - 1
    for child, edge in self.nxgraph[task].items():
      if child.name != "end":
        right_boundary = min(right_boundary, self.scheduling[child] - 1)
    for parent in self.nxgraph.pred[task]:
      if parent.name != "root":
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
    self.score = -1
    task = random.choice(self.tasks)
    self.matching[task] = random.choice(self.hosts)

  def make_task_order(self):
    res = copy.copy(self.tasks)
    for task in self.scheduling:
      res[self.scheduling[task]] = task
    return res

  def make_final_task_order(self):
    res = copy.copy(self.tasks)
    res.append(res[0])
    res.append(res[0])
    for task in self.scheduling:
      for parent in self.nxgraph.pred[task]:
        if parent.name == "root":
          res[0] = parent
    for task in self.scheduling:
      res[self.scheduling[task] + 1] = task
    for task in self.scheduling:
      for child, edge in self.nxgraph[task].items():
        if child.name == "end":
          res[len(res) - 1] = child
    return res


  def matching_crossover(self, gen):
    self.score = -1
    gen.score = -1
    cut_off = random.randint(0, len(self.tasks) - 1)
    for task in self.tasks:
      if cut_off == 0:
        gen.matching[task], self.matching[task] = self.matching[task], gen.matching[task]
      else:
        cut_off -= 1


  def scheduling_crossover(self, gen):
    self.score = -1
    gen.score = -1
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

  def __eq__(self, other):
    return self.scheduling == other.scheduling and self.matching == other.matching



def _make_final_state(simulation, gen):
  state = cscheduling.SchedulerState(simulation)
  platform_model = cscheduling.PlatformModel(simulation)
  order = gen.make_final_task_order()
  for i in range(len(order)):
    task_to_schedule = order[i]
    # print (task_to_schedule.name, len( gen.nxgraph.pred[task_to_schedule]))
    if cscheduling.try_schedule_boundary_task(task_to_schedule, platform_model, state) == False:
      host_to_schedule = gen.matching[task_to_schedule]
      # print(host_to_schedule.name)
      est = platform_model.est(host_to_schedule, dict(gen.nxgraph.pred[task_to_schedule]), state)
      eet = platform_model.eet(task_to_schedule, host_to_schedule)
      timesheet = state.timetable[host_to_schedule]
      pos, start, finish = cscheduling.timesheet_insertion(timesheet, est, eet)
      state.update(task_to_schedule, host_to_schedule, pos, start, finish)
  return state

def _evaluation(simulation, gen):
  state = _make_final_state(simulation, gen)
  expected_makespan = max([state["ect"] for state in state.task_states.values()])
  return expected_makespan


def _return_non_random_schedules(simulation):
  res = []
  classes = [DLS(simulation), HCPT(simulation), HEFT(simulation),
             Lookahead(simulation), PEFT(simulation)]
  for cl in classes:
    res.append(cl.get_schedule(simulation))
  return res


def _weighted_choice(weights):
  sum_weights = sum(weights)
  r = random.uniform(0, sum_weights)
  upto = 0
  for idx in range(len(weights)):
    w = weights[idx]
    if upto + w >= r:
      return idx
    upto += w
  assert False, "Shouldn't get here"


def _make_chromosome_copy(gen):
  new_gen = Chromosome(gen.tasks, gen.hosts, gen.nxgraph)
  new_gen.score = gen.score
  new_gen.scheduling = copy.copy(gen.scheduling)
  new_gen.matching = copy.copy(gen.matching)
  return new_gen


class GA(StaticScheduler):
  """
  Genetic-Algorithm-Based Approach

  """

  def get_schedule(self, simulation):
    """
    Overridden.
    """
    random.seed(12345)
    POPULATION_SIZE = 50
    NUMBER_OF_ITERATION = 100
    RANK_CONSTANT = 1.1
    CROSSOVER_PROBABILITY = 1.0
    MUTATION_PROBABILITY = 0.8
    SILENT_MODE = True

    nxgraph = simulation.get_task_graph()
    platform_model = cscheduling.PlatformModel(simulation)
    state = cscheduling.SchedulerState(simulation)

    topological_order = list(networkx.topological_sort(nxgraph))
    topological_order_without_end_and_root = []
    for task in topological_order:
      if task.name == "root" or task.name == "end":
        continue
      topological_order_without_end_and_root.append(task)

    hosts_without_master = []
    for host in simulation.hosts:
      if host.name == "master":
        continue
      hosts_without_master.append(host)

    schedules = _return_non_random_schedules(simulation)

    popultaion = []
    for schedule in schedules:
      gen = Chromosome(topological_order_without_end_and_root, hosts_without_master, nxgraph)
      gen.copy_schedule(schedule)
      popultaion.append(gen)

    for i in range(POPULATION_SIZE - len(popultaion)):
      gen = Chromosome(topological_order_without_end_and_root, hosts_without_master, nxgraph)
      popultaion.append(gen)

    for epoch in range(NUMBER_OF_ITERATION):
      if SILENT_MODE == False:
        print ("Starting epoсh", epoch, ":")

      # Evaluation
      for gen in popultaion:
        if gen.score == -1:
          ans = _evaluation(simulation, gen)
          gen.score = ans

      # Selection
      weights = []
      population_size = len(popultaion)
      for i in range(population_size):
        weights.append(RANK_CONSTANT ** (population_size - i - 1) * (RANK_CONSTANT - 1))
      popultaion = sorted(popultaion, key = lambda t: t.score)
      if SILENT_MODE == False:
        print ("Best score", popultaion[0].score)
      new_population = []
      for idx in range(POPULATION_SIZE):
        index_to_push = _weighted_choice(weights)
        # print(index_to_push)
        new_population.append(_make_chromosome_copy(popultaion[index_to_push]))
      popultaion = new_population

      # Crossover
      for idx in range(POPULATION_SIZE // 2):
        prob = random.uniform(0, 1)
        if prob < CROSSOVER_PROBABILITY:
          gen1 = _make_chromosome_copy(random.choice(popultaion))
          gen2 = _make_chromosome_copy(random.choice(popultaion))
          gen1.matching_crossover(gen2)
          gen1.scheduling_crossover(gen2)
          popultaion.append(gen1)
          popultaion.append(gen2)

      # Mutation
      for idx in range(POPULATION_SIZE):
        prob = random.uniform(0, 1)
        if prob < MUTATION_PROBABILITY:
          gen = _make_chromosome_copy(random.choice(popultaion))
          gen.matching_mutation()
          gen.scheduling_mutation()
          popultaion.append(gen)

    print("Final population: ")
    for gen in popultaion:
      ans = _evaluation(simulation, gen)
      gen.score = ans
      if SILENT_MODE == False:
        print("Scores:", ans)
    popultaion = sorted(popultaion, key = lambda t: t.score)
    final_state = _make_final_state(simulation, popultaion[0])
    return final_state.schedule, popultaion[0].score

