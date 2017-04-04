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

from ... import cscheduling
from ..scheduler import StaticScheduler


class GA(StaticScheduler):
    """
    Genetic-Algorithm-Based Approach

    """

    class Chromosome:

        class FakeScheduler(StaticScheduler):
            def __init__(self, schedule):
                self.schedule = schedule

            def get_schedule(self, simulation):
                return self.schedule


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
            for task in self.tasks:
                if cut_off == 0:
                    temp = gen.matching[task]
                    gen.matching[task] = self.matching[task]
                    self.matching[task] = temp
                else:
                    cut_off -= 1

        def scheduling_crossover(self, gen):
            cut_off = random.randint(0, len(self.tasks) - 1)
            cut_off_temp = cut_off
            order = self.make_task_order()
            first_res = {}
            for task in self.tasks:
                if cut_off_temp == 0:
                    break
                else:
                    first_res[task] = gen.scheduling[task]
                    cut_off_temp -= 1
            id = cut_off
            for task in order:
                if task not in first_res:
                    first_res[task] = id
                    id += 1

            cut_off_temp = cut_off
            order = gen.make_task_order()
            second_res = {}
            for task in self.tasks:
                if cut_off_temp == 0:
                    break
                else:
                    second_res[task] = self.scheduling[task]
                    cut_off_temp -= 1
            id = cut_off
            for task in order:
                if task not in second_res:
                    second_res[task] = id
                    id += 1

            gen.scheduling = first_res
            self.scheduling = second_res

        def evaluation(self):
            schedule = {host: [] for host in self.hosts}
            order = self.make_task_order()
            for task in order:
                schedule[self.matching[task]].append(task)
            fake_schedule = FakeScheduler(schedule)

            return 0

        def __eq__(self, other):
            return self.scheduling == other.scheduling and self.matching == other.matching





    def get_schedule(self, simulation):
        """
        Overridden.
        """
        nxgraph = simulation.get_task_graph()
        platform_model = cscheduling.PlatformModel(simulation)
        state = cscheduling.SchedulerState(simulation)

        mean_speed = platform_model.mean_speed
        aec, sl = self.get_tasks_sl_aec(nxgraph, platform_model)
        # unreal dynamic level - used to mark deleted on not set values in a queue
        unreal_dl = 1 + max(sl.items(), key=operator.itemgetter(1))[1] + max(aec.items(), key=operator.itemgetter(1))[1]
        dl = {host: {task: unreal_dl for task in nxgraph} for host in simulation.hosts}
        undone_parents = {task: len(nxgraph.pred[task]) for task in nxgraph}
        waiting_tasks = set(nxgraph)
        queue_tasks = set()
        for task in nxgraph:
            if undone_parents[task] == 0:
                for host in simulation.hosts:
                    dl[host][task] = sl[task] + (task.amount / mean_speed - task.amount / host.speed)
                waiting_tasks.remove(task)
                queue_tasks.add(task)

        for iterations in range(len(nxgraph)):
            cur_max = unreal_dl
            task_to_schedule = -1
            host_to_schedule = -1
            for host in simulation.hosts:
                for task in queue_tasks:
                    if dl[host][task] == unreal_dl:
                        continue
                    if cur_max == unreal_dl or dl[host][task] > cur_max:
                        cur_max = dl[host][task]
                        host_to_schedule = host
                        task_to_schedule = task

            assert (cur_max != unreal_dl)

            if cscheduling.try_schedule_boundary_task(task_to_schedule, platform_model, state) == False:
                est = platform_model.est(host_to_schedule, nxgraph.pred[task_to_schedule], state)
                eet = platform_model.eet(task_to_schedule, host_to_schedule)
                timesheet = state.timetable[host_to_schedule]
                pos, start, finish = cscheduling.timesheet_insertion(timesheet, est, eet)
                state.update(task_to_schedule, host_to_schedule, pos, start, finish)

            new_tasks = set()
            for child, edge in nxgraph[task_to_schedule].items():
                undone_parents[child] -= 1
                if undone_parents[child] == 0:
                    new_tasks.add(child)
                    for host in simulation.hosts:
                        dl[host][child] = self.calculate_dl(nxgraph, platform_model, state, sl, aec, child, host)

            for host in simulation.hosts:
                dl[host][task_to_schedule] = unreal_dl

            queue_tasks.remove(task_to_schedule)

            for task in queue_tasks:
                if undone_parents[task] == 0:
                    dl[host_to_schedule][task] = self.calculate_dl(nxgraph, platform_model, state, sl, aec, task,
                                                                   host_to_schedule)

            for task in new_tasks:
                waiting_tasks.remove(task)
                queue_tasks.add(task)

        return state.schedule

    @classmethod
    def calculate_dl(cls, nxgraph, platform_model, state, sl, aec, task, host):
        est = platform_model.est(host, nxgraph.pred[task], state)
        eet = platform_model.eet(task, host)
        timesheet = state.timetable[host]
        pos, start, finish = cscheduling.timesheet_insertion(timesheet, est, eet)
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
        topological_order = networkx.topological_sort(nxgraph, reverse=True)

        # Average execution cost
        aec = {task: float(task.amount) / mean_speed for task in nxgraph}

        sl = {task: aec[task] for task in nxgraph}

        # Static Level
        for task in topological_order:
            for parent in nxgraph.pred[task]:
                sl[parent] = max(sl[parent], sl[task] + aec[parent])

        return aec, sl

