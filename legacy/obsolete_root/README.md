已从仓库根目录移走的旧脚本。

这里主要放两类文件：

- 已废弃的 PPO / BC 兼容入口与包装层
- 早期训练或迁移期脚本，不再属于当前 transformer 主线

当前主线请优先查看仓库根目录中的 `run_iteration.py`、`loop_train.py`、`train_mcts_nn.py`、`battle.py`、`server.py` 等脚本。

建议从仓库根目录执行这里的旧脚本，例如 `python legacy/obsolete_root/train.py`，以避免相对路径导入或数据文件定位出错。
