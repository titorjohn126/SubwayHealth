from mmengine.evaluator import BaseMetric, Evaluator
from mmengine.evaluator.metric import DumpResults, _to_cpu, dump
from mmcls.registry import METRICS, EVALUATORS

import torch

@EVALUATORS.register_module()
class MyEvaluator(Evaluator):
    def process(self, data_samples, data_batch=None):
        for metric in self.metrics:
            metric.process(data_batch, data_samples)

@METRICS.register_module()
class MyAcc(BaseMetric):
    def __init__(self, collect_device='cpu', prefix=None) -> None:
        super().__init__(collect_device, prefix)

    def process(self, data_batch, data_samples) -> None:
        """ Store label & data_samples in self.results
        data_batch: one iter from dataloader
        data_samples: outputs of model prediction (B,)
        """
        _, label = data_batch   # label (B,)
        self.results.append((label, data_samples))


    def compute_metrics(self, results: list) -> dict:
        ret = {}
        total, pos = 0, 0
        for label, pred in results:
            total += len(label)
            pos += (label == pred).sum()
        acc = pos / max(total, 1)
        ret['acc'] = acc
        return ret

@METRICS.register_module()
class MyDumpResults(DumpResults):
    """ Dump cls results as a dict(label, pred)
    """
    def process(self, data_batch, predictions) -> None:
        _, label = data_batch
        item = (label, predictions)
        self.results.append(item)

    def compute_metrics(self, results: list) -> dict:
        label, pred = list(zip(*results))
        label, pred = map(torch.cat, (label, pred))
        output = dict(label=label, pred=pred)
        dump(_to_cpu(output), self.out_file_path)
        return {}
