import logging

import jax

jax.config.update("jax_enable_x64", True)

# global level logging formatting
logging.basicConfig(
    format="%(asctime)s [%(name)s/%(levelname)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)

# disable jax XLA bridge logger
logging.getLogger("jax._src.xla_bridge").setLevel(logging.ERROR)
