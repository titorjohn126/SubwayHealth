from mmcls.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class LogModelHook(Hook):
    def before_run(self, runner) -> None:
        model = runner.model.module if runner.distributed \
                                  else runner.model
        runner.logger.info(model)