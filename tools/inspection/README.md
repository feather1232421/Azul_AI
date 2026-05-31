一次性查看、调试、数据检查脚本。

这些文件通常用于：

- 快速查看旧数据集或 checkpoint 内容
- 做临时 benchmark 或命令行试玩
- 排查历史实验数据

它们不属于当前 transformer 训练/对战主线，因此已从仓库根目录移出。

建议从仓库根目录执行这里的脚本，例如 `python tools/inspection/test.py`，否则部分脚本里的相对路径可能会失效。
