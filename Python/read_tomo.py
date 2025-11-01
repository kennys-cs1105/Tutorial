import tomllib
from pprint import pp

with open("config.toml", "br") as f:
    config = tomllib.load(f)

pp(config)
