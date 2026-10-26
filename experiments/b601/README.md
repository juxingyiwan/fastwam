# FastWAM B601 adapter

[完整中文操作指南](../../docs/rebot.zh-CN.md) · [English guide](../../docs/rebot.md)

- [Data and coordinate contract](docs/data.md)
- [Machine-local deployment template](configs/local.example.yml)
- [Training paths template](configs/paths.env.example)
- [Hardware dependency overlay](requirements-deploy.txt)
- [Homing implementation and historical changes](NATIVE_HOMING.md)
- [Hardware-free regression tests](tests)

Run `bash experiments/b601/local.sh --help` from the repository root in the selected environment. Machine configuration and experiment output stay outside the checkout. The maintained single-task training entry is `train_single_task.sh`; site-specific background delivery jobs are not part of this submission tree.
