# FastWAM: LIBERO, RoboTwin and ReBot

[中文](README.zh-CN.md) · [▶ Video showcase](https://juxingyiwan.github.io/fastwam/) · [Upstream FastWAM](https://github.com/yuantianyuan01/FastWAM) · [Contributing](CONTRIBUTING.md) · [All CI checks](https://github.com/juxingyiwan/fastwam/actions)

[![Source and CPU regressions](https://github.com/juxingyiwan/fastwam/actions/workflows/cpu.yml/badge.svg?branch=main)](https://github.com/juxingyiwan/fastwam/actions/workflows/cpu.yml)

**Train FastWAM policies, evaluate them in simulation and deploy them on ReBot / B601.**

This submission tree is based on upstream FastWAM `7faa71108368fbb3b6885649f112af607427a2d4` and the audited deployment source. It preserves the deployed model, control, homing and RTC implementations. Machine-specific orchestration and generated artifacts are kept outside the project. Source identities are recorded in [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json).

[Get started](#workflows) · [Robot demonstrations](#rebot-demos) · [Development and provenance](#development)

<a id="rebot-demos"></a>

## ReBot / B601: FastWAM with RTC

Three real-robot task demonstrations with **RTC enabled**. The silent animations show each complete demonstration at its original speed. Click a preview to play the original 720p video with sound and fullscreen controls. The headings show the full task instructions.

[▶ Video showcase](https://juxingyiwan.github.io/fastwam/)

| `Grasp the cube to the bowl` | `Grasp the pen to the box` | `Grasp the cube into the closed top drawer` |
| --- | --- | --- |
| [![FastWAM with RTC: Grasp the cube to the bowl](assets/demos/fastwam-cube-rtc.gif)](https://juxingyiwan.github.io/fastwam/#cube) | [![FastWAM with RTC: Grasp the pen to the box](assets/demos/fastwam-pen-rtc.gif)](https://juxingyiwan.github.io/fastwam/#pen) | [![FastWAM with RTC: Grasp the cube into the closed top drawer](assets/demos/fastwam-drawer-rtc.gif)](https://juxingyiwan.github.io/fastwam/#drawer) |
| [▶ Watch video · 18.7 s](https://juxingyiwan.github.io/fastwam/#cube) | [▶ Watch video · 11.4 s](https://juxingyiwan.github.io/fastwam/#pen) | [▶ Watch video · 17.4 s](https://juxingyiwan.github.io/fastwam/#drawer) |

[Robot deployment guide](docs/rebot.md) · [Video metadata](assets/demos/manifest.json)

<a id="workflows"></a>

## Choose a workflow

| Goal | Guide | Main entry |
| --- | --- | --- |
| LIBERO training and evaluation | [English](docs/libero.md) · [中文](docs/libero.zh-CN.md) | `experiments/libero/run_libero_manager.py` |
| RoboTwin training and evaluation | [English](docs/robotwin.md) · [中文](docs/robotwin.zh-CN.md) | `experiments/robotwin/run_robotwin_manager.py` |
| B601 data, training and deployment | [English](docs/rebot.md) · [中文](docs/rebot.zh-CN.md) | `experiments/b601/local.sh` |
| Environment selection | [Environment profiles](environments/README.md) | Separate training, robot and CPU-test environments |

Run documented commands from the checkout root. Models, datasets, simulator assets, calibrated device paths and robot configuration are supplied separately. GPU training and inference require an appropriate Linux/CUDA environment.

```bash
git rev-parse HEAD
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## Repository layout

```text
src/fastwam/            model, trainer, datasets, EE-delta and RTC primitives
experiments/b601/      robot adapter, local runner, model export and CPU tests
experiments/libero/    benchmark runners and optional visual-only TTT
experiments/robotwin/  benchmark runners and policy adapter
configs/              public Hydra configurations
scripts/              training and data-preparation commands
third_party/RoboTwin/  vendored simulator source; large assets are external
docs/                 scenario guides
environments/         separate dependency and validation profiles
```

## Real-robot contract

The adapter uses a seven-dimensional base-frame end-effector delta action: translation, rotation vector and one gripper channel. Each decoded action is anchored to fresh measured feedback. The default configuration uses synchronous execution; guided RTC is an explicit option. See the [data contract](experiments/b601/docs/data.md), [configuration template](experiments/b601/configs/local.example.yml) and [homing implementation notes](experiments/b601/NATIVE_HOMING.md).

Start with `models` and `config`, then an offline `check` using a compatible bundle. Hardware preflight and robot runs require an attended robot session. CPU checks do not validate a physical trajectory or task success.

<a id="development"></a>

## Development and provenance

```bash
python scripts/check_submission.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH="$PWD/src:$PWD" \
  python -m pytest -q experiments/b601/tests experiments/libero/visual_ttt
```

The [contribution guide](CONTRIBUTING.md) and [CPU workflow](.github/workflows/cpu.yml) define the checks. The [validation note](docs/validation.md) records what was actually run. This project does not claim new benchmark scores or robot success rates from a source synchronization.

Original upstream documentation is preserved at [README.upstream.md](README.upstream.md) and [README.upstream.zh-CN.md](README.upstream.zh-CN.md). The documentation organization was informed by [juxingyiwan/wall-x](https://github.com/juxingyiwan/wall-x/tree/2625a5bcf6066c873853bf9554b1041157d0f65c).

FastWAM retains its [MIT license](LICENSE). The adapted B601 runtime retains its [Apache-2.0 license](experiments/b601/runtime/LICENSE). Other components retain their own licenses and headers; see [NOTICE.txt](NOTICE.txt).
