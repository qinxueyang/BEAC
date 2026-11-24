"""BEAC 论文主方法训练入口。

主训练实现位于 ``beac_full_model.train_impl``，公开运行入口是
``scripts/run_beac_full_model.sh``。
"""

from . import train_impl as _impl

build_parser = _impl.build_parser
configure = _impl.configure
main = _impl.main
ABLATION_DESCRIPTIONS = _impl.ABLATION_DESCRIPTIONS

__all__ = ["main", "build_parser", "configure", "ABLATION_DESCRIPTIONS"]
