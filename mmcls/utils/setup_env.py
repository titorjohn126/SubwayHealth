# Copyright (c) OpenMMLab. All rights reserved.
import datetime
import warnings

from mmengine import DefaultScope


def register_all_modules(init_default_scope: bool = True) -> None:
    """ The param `init_default_scope` is a placeholder, to be
    compatible with mmengine's usage. And here we do not structures
    defined in mmlab.
    """
    import mmcls.datasets  # noqa: F401,F403
    import mmcls.engine  # noqa: F401,F403
    import mmcls.evaluation  # noqa: F401,F403
    import mmcls.models  # noqa: F401,F403
    # import mmcls.structures  # noqa: F401,F403
    import mmcls.visualization  # noqa: F401,F403