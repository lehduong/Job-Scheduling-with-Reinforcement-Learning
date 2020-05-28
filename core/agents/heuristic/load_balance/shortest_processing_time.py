import numpy as np


class ShortestProcessingTimeAgent(object):
    def __init__(self):
        pass

    def get_action(self, state):
        workers, _, _ = state

        min_time_idx = None
        min_time = np.inf

        for i in range(len(workers)):
            worker = workers[i]
            work = np.sum([j.size for j in worker.queue])
            remain_time = work / worker.service_rate
            if remain_time < min_time:
                min_time_idx = i
                min_time = remain_time

        return min_time_idx
