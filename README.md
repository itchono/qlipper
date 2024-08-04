<div align="center">

[![GitHub tag](https://img.shields.io/github/tag/itchono/qlipper.svg)](https://github.com/itchono/qlipper/tags)
![Python 3.9+](https://img.shields.io/badge/Python-3.9+-1081c1?logo=python)

# qlipper: A Q-Law Implementation in Python

</div>

```plaintext
          ___
   ____ _/ (_)___  ____  ___  _____
  / __ `/ / / __ \/ __ \/ _ \/ ___/
 / /_/ / / / /_/ / /_/ /  __/ /
 \__, /_/_/ .___/ .___/\___/_/
   /_/   /_/   /_/
```

A moderately fast implementation of my solar sail guidance law, QUAIL, developed for my 4th year undergraduate thesis in aerospace engineering.

Extending the research done during my undergrad thesis to greater heights using a more robustly written and well-developed simulator.

Qlipper also represents my quest to learn JAX and make an autograd-based implementation of the Q-Law, as a spiritual successor to [Star Sailor](https://github.com/itchono/star-sailor).

## Project Plans

* Milestone 1: Mirrored implementation of existing [SLyGA](https://github.com/itchono/SLyGA) repository, except all in Python (DONE)
* Milestone 2: Getting autograd and jit to work so code becomes fast (DONE)
* Milestone 3: Expansion of the Q-Law to include 6-element static targeting
* Milestone 4: Multibody transfers

## Showcase

Example transfer from my paper.
![image](https://github.com/user-attachments/assets/69c0a23a-4525-49a5-b44b-8059276ea27f)
![image](https://github.com/user-attachments/assets/f72b09dc-9dc0-4bb7-b4c3-4ffe047a128b)

## Install

Python 3.9+ is required.

1. Clone the repo
2. `cd` to repo root
3. (Optional) [Create a virtual environment](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment)
4. Run `pip install -e .` to install qlipper as a package. This automatically installs required dependencies.

### Optional Dependencies

* `pip install -e .[dev]` - for unit testing

## Tests

Run `pytest` from the root folder to run tests.
