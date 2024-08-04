import sys
import random
from typing import Tuple, Any, Annotated, get_type_hints

import numpy as np

from tinygrad import Tensor, dtypes
from tinygrad.dtype import DType

from hypothesis.extra.numpy import arrays
from hypothesis.strategies import integers, composite
from hypothesis import given, settings

from chalk import hcat, vcat, text, rectangle, set_svg_height, vstrut
from colour import Color
from IPython.display import display, SVG


class TensorType(Tensor):
  # def __new__(cls, *args, **kwargs):
  #     raise RuntimeError(f"Class {cls.__name__} cannot be instantiated.")

  def __class_getitem__(cls, args: Tuple[Any, ...]):
    if not isinstance(args, tuple):
      args = (args,)
    info = {}
    for arg in args:
      if (
        isinstance(arg, list)
        and info.get("shape") is None
        and all(isinstance(a, (str, int)) for a in arg)
      ):
        info["shape"] = tuple(arg)
      elif isinstance(arg, DType):
        info["dtype"] = arg
    return Annotated[Tensor, info]


def color(v):
  d = rectangle(1, 1)
  if v == 0:
    return d
  elif v > 0:
    return d.fill_color(Color("orange")).fill_opacity(0.4 + 0.6 * (v / 10))
  elif v < 0:
    return d.fill_color(Color("blue")).fill_opacity(0.4 + 0.6 * (abs(v) / 10))


def draw_matrix(mat):
  return vcat(
    (hcat((color(v) for j, v in enumerate(inner))) for i, inner in enumerate(mat))
  )


def grid(diagrams):
  mhs = [0] * 100
  mws = [0] * 100
  for i, row in enumerate(diagrams):
    for j, col in enumerate(row):
      env = col.get_envelope()
      mhs[i] = max(env.height, mhs[i])
      mws[j] = max(mws[j], env.width)
  return vcat(
    [
      hcat(
        [
          col.center_xy().with_envelope(rectangle(mws[j], mhs[i]))
          for j, col in enumerate(row)
        ],
        1.0,
      )
      for i, row in enumerate(diagrams)
    ],
    1.0,
  )


def draw_example(data):
  name = data["name"]
  keys = list(data["vals"][0].keys())
  # cols = [[vstrut(0)] + [vstrut(0.5) / text(f"Ex. {i}", 0.5).fill_color(Color("black")).line_width(0.0) / vstrut(0.5) for i in range(len(data["vals"]))]]
  cols = []
  for k in keys:
    mat = [
      (
        vstrut(0.5)
        / text(k, 0.5).fill_color(Color("black")).line_width(0.0)
        / vstrut(0.5)
      )
    ]
    for ex in data["vals"]:
      v2 = ex[k]
      mat.append(draw_matrix(v2))
    cols.append(mat)

  full = grid(cols)

  full = (
    vstrut(1)
    / text(name, 0.75).fill_color(Color("black")).line_width(0)
    / vstrut(1)
    / full.center_xy()
  )
  full = full.pad(1.2).center_xy()
  env = full.get_envelope()
  set_svg_height(50 * env.height)
  set_svg_height(300)
  return rectangle(env.width, env.height).fill_color(Color("white")) + full


def draw_examples(name, examples):
  data = {
    "name": name,
    "vals": [
      {k: [v.tolist()] if len(v.shape) == 1 else v.tolist() for k, v in example.items()}
      for example in examples
    ],
  }
  return draw_example(data)


tinygrad_to_numpy_dtype = {
  dtypes.bool: np.bool,
  dtypes.uint8: np.uint8,
  dtypes.int8: np.int8,
  dtypes.int16: np.int16,
  dtypes.int32: np.int32,
  dtypes.int64: np.int64,
  dtypes.float16: np.float16,
  dtypes.float32: np.float32,
  dtypes.float64: np.float64,
}


@composite
def spec(draw, x, min_size=1):
  # Get the type hints.
  if sys.version_info >= (3, 9):
    gth = get_type_hints(x, include_extras=True)
  else:
    gth = get_type_hints(x)

  # Collect all the dimension names.
  names = set()
  for k in gth:
    if not hasattr(gth[k], "__metadata__"):
      continue
    dims = gth[k].__metadata__[0]["shape"]
    names.update([d for d in dims if isinstance(d, str)])
  names = list(names)

  # draw sizes for each dim.
  size = integers(min_value=min_size, max_value=5)
  arr = draw(
    arrays(shape=(len(names),), unique=True, elements=size, dtype=np.int32)
  ).tolist()
  sizes = dict(zip(names, arr))
  for n in list(sizes.keys()):
    if "*" in n or "+" in n or "-" in n or "//" in n:
      i, op, j = n.split()
      i_val = i if i.isdigit() else sizes[i]
      j_val = j if j.isdigit() else sizes[j]
      sizes[n] = eval("{}{}{}".format(i_val, op, j_val))

  # Create tensors for each size.
  ret = {}
  for k in gth:
    if not hasattr(gth[k], "__metadata__"):
      continue
    shape = tuple(
      [sizes[d] if isinstance(d, str) else d for d in gth[k].__metadata__[0]["shape"]]
    )
    dtype = (
      tinygrad_to_numpy_dtype[gth[k].__metadata__[0]["dtype"]]
      if hasattr(gth[k].__metadata__[0], "dtype")
      else np.int32
    )
    ret[k] = draw(
      arrays(
        shape=shape,
        dtype=dtype,
        elements=integers(min_value=-5, max_value=5) if dtype is int else None,
        unique=False,
      )
    )
    ret[k] = np.nan_to_num(ret[k], nan=0, neginf=0, posinf=0)

  ret["return"][:] = 0
  return ret, sizes


def make_test(name, problem, problem_spec, add_sizes=[], constraint=lambda d: d):
  examples = []
  for i in range(3):
    example, sizes = spec(problem, 3).example()
    example = constraint(example)
    out = example["return"].tolist()
    del example["return"]
    problem_spec(*example.values(), out)

    for size in add_sizes:
      example[size] = sizes[size]

    yours = None
    try:
      yours = problem(
        *map(lambda v: v if isinstance(v, int) else Tensor(v), example.values())
      )

    except NotImplementedError:
      pass
    for size in add_sizes:
      del example[size]
    example["target"] = Tensor(out)
    if yours is not None:
      example["yours"] = yours
    examples.append(example)

  diagram = draw_examples(name, examples)
  display(SVG(diagram._repr_svg_()))

  @given(spec(problem))
  @settings(deadline=None)
  def test_problem(d):
    d, sizes = d
    d = constraint(d)
    out = d["return"].tolist()
    del d["return"]
    problem_spec(*d.values(), out)
    for size in add_sizes:
      d[size] = sizes[size]

    out = Tensor(out)
    out2 = problem(
      *map(lambda v: v if isinstance(v, int) else Tensor(v), d.values())
    )._broadcast_to(out.shape)
    assert np.allclose(out.numpy(), out2.numpy()), (
      "Two tensors are not equal\n Spec: \n\t%s \n\t%s" % (out, out2)
    )

  return test_problem


def run_test(fn):
  fn()
  # Generate a random puppy video if you are correct.
  print("Correct!")
  from IPython.display import HTML

  pups = [
    "2m78jPG",
    "pn1e9TO",
    "MQCIwzT",
    "udLK6FS",
    "ZNem5o3",
    "DS2IZ6K",
    "aydRUz8",
    "MVUdQYK",
    "kLvno0p",
    "wScLiVz",
    "Z0TII8i",
    "F1SChho",
    "9hRi2jN",
    "lvzRF3W",
    "fqHxOGI",
    "1xeUYme",
    "6tVqKyM",
    "CCxZ6Wr",
    "lMW0OPQ",
    "wHVpHVG",
    "Wj2PGRl",
    "HlaTE8H",
    "k5jALH0",
    "3V37Hqr",
    "Eq2uMTA",
    "Vy9JShx",
    "g9I2ZmK",
    "Nu4RH7f",
    "sWp0Dqd",
    "bRKfspn",
    "qawCMl5",
    "2F6j2B4",
    "fiJxCVA",
    "pCAIlxD",
    "zJx2skh",
    "2Gdl1u7",
    "aJJAY4c",
    "ros6RLC",
    "DKLBJh7",
    "eyxH0Wc",
    "rJEkEw4",
  ]
  return HTML(
    """
    <video alt="test" controls autoplay=1>
        <source src="https://openpuppies.com/mp4/%s.mp4"  type="video/mp4">
    </video>
    """
    % (random.sample(pups, 1)[0])
  )
