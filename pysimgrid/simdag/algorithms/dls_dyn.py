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

import networkx
import numpy
import operator

from .. import scheduler
from ... import csimdag
from ... import cscheduling


class DynamicDLS(scheduler.DynamicScheduler):

  def prepare(self, simulation):
    for h in simulation.hosts:
      h.data = {
        "est": 0.
      }
    master_hosts = simulation.hosts.by_prop("name", self.MASTER_HOST_NAME)
    self._master_host = master_hosts[0] if master_hosts else None
    if self._master_host:
      for task in simulation.tasks.by_func(lambda t: t.name in self.BOUNDARY_TASKS):
        task.schedule(self._master_host)
    self._exec_hosts = simulation.hosts.by_prop("name", self.MASTER_HOST_NAME, True)
    self._started_tasks = set()
    self._estimate_cache = {}

    nxgraph = simulation.get_task_graph()
    platform_model = cscheduling.PlatformModel(simulation)
    state = cscheduling.SchedulerState(simulation)

    mean_speed = platform_model.mean_speed
    self._aec, self._sl = self.get_tasks_sl_aec(nxgraph, platform_model)
    # unreal dynamic level - used to mark deleted on not set values in a queue
    self._unreal_dl = 1 + max(self._sl.items(), key=operator.itemgetter(1))[1] + max(self._aec.items(), key=operator.itemgetter(1))[1]
    self._dl = {host: {task: self._unreal_dl for task in nxgraph} for host in simulation.hosts}
    self._undone_parents = {task: len(nxgraph.pred[task]) for task in nxgraph}
    waiting_tasks = set(nxgraph)
    queue_tasks = set()
    for task in nxgraph:
      if self._undone_parents[task] == 0:
        for host in simulation.hosts:
          self._dl[host][task] = self._sl[task] + (task.amount / mean_speed - task.amount / host.speed)
        waiting_tasks.remove(task)
        queue_tasks.add(task)

### Schedule

  def schedule(self, simulation, changed):
    clock = simulation.clock
    free_hosts = set(self._exec_hosts)
    for task in simulation.tasks[csimdag.TaskState.TASK_STATE_RUNNING, csimdag.TaskState.TASK_STATE_SCHEDULED]:
      host = task.hosts[0]
      free_hosts.discard(host)
      if task.start_time > 0 and task not in self._started_tasks:
        self._started_tasks.add(task)
        host.data["est"] = task.start_time + task.get_eet(host)
    host_est = {}
    for h in self._exec_hosts:
      host_est[h] = h.data["est"]

    queue_tasks = set(simulation.tasks[csimdag.TaskState.TASK_STATE_SCHEDULABLE])
    for task in queue_tasks:
      for host in self._exec_hosts:
        self._dl[host][task] = self.calculate_dl(self._sl, self._aec,
                                                 task, host, max(clock, host.data["est"]))

    nxgraph = simulation.get_task_graph()
    while len(queue_tasks) != 0:
      cur_max = self._unreal_dl
      task_to_schedule = -1
      host_to_schedule = -1
      for host in simulation.hosts:
        if cscheduling.is_master_host(host):
          continue
        for task in queue_tasks:
          if self._dl[host][task] == self._unreal_dl:
            continue
          if cur_max == self._unreal_dl or self._dl[host][task] > cur_max:
            cur_max = self._dl[host][task]
            host_to_schedule = host
            task_to_schedule = task

      assert (cur_max != self._unreal_dl)

      host_est[host_to_schedule] = self.get_ect(host_est[host_to_schedule], clock,
                                                task_to_schedule, host_to_schedule)

      new_tasks = set()
      if host_to_schedule in free_hosts:
        task_to_schedule.schedule(host_to_schedule)
        # logging.info("%s -> %s" % (task.name, target_host.name))
        host_to_schedule.data["est"] = self.get_ect(host_to_schedule.data["est"], clock, task_to_schedule, host_to_schedule)
        free_hosts.remove(host_to_schedule)
        if len(free_hosts) == 0:
          break


      queue_tasks.remove(task_to_schedule)

      for task in queue_tasks:
        self._dl[host_to_schedule][task] = self.calculate_dl(self._sl, self._aec, task,
                                                             host_to_schedule, max(clock, host_est[host_to_schedule]))

      #for task in new_tasks:
      #queue_tasks.add(task)

  def get_ect(self, est, clock, task, host):
    if (task, host) in self._estimate_cache:
      task_time = self._estimate_cache[(task, host)]
    else:
      parent_connections = [p for p in task.parents if p.kind == csimdag.TaskKind.TASK_KIND_COMM_E2E]
      comm_times = [conn.get_ecomt(conn.parents[0].hosts[0], host) for conn in parent_connections]
      task_time = (max(comm_times) if comm_times else 0.) + task.get_eet(host)
      self._estimate_cache[(task, host)] = task_time
    return max(est, clock) + task_time

  @classmethod
  def calculate_dl(cls, sl, aec, task, host, start):
    return sl[task] + (aec[task] - task.amount / host.speed) - start

  @classmethod
  def get_tasks_sl_aec(cls, nxgraph, platform_model):
    """
    Return Average Execution Cost and Static Level for every task.

    Args:
      nxgraph: full task graph as networkx.DiGraph
      platform_model: cscheduling.PlatformModel object

    Returns:
        aec: task->aec_value
        sl: task->static_level_value
    """
    mean_speed = platform_model.mean_speed
    topological_order = list(reversed(list(networkx.topological_sort(nxgraph))))

    # Average execution cost
    aec = {task: float(task.amount) / mean_speed for task in nxgraph}

    sl = {task: aec[task] for task in nxgraph}

    # Static Level
    for task in topological_order:
      for parent in nxgraph.pred[task]:
        sl[parent] = max(sl[parent], sl[task] + aec[parent])

    return aec, sl
