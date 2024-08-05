from lib import TinyType as TT
from tinygrad import Tensor, dtypes


# TODO: tinygrad should support these operations
def _fd(a: Tensor, b: Tensor) -> Tensor:
  """floor division"""
  return (a / b).floor().cast(dtypes.int)

def _m(a: Tensor, b: Tensor) -> Tensor:
  """modulus"""
  return a - _fd(a, b) * b


def arange(i: int):
  "Use this function to replace a for-loop."
  return Tensor(list(range(i)))


def where(q, a, b):
  "Use this function to replace an if-statement."
  return q * a + q.logical_not() * b


def ones(i: int) -> TT[["i"]]:
  return (arange(i) >= 0) * 1


def sum(a: TT[["i"]]) -> TT[[1]]:
  # `[:, None]` is necessary to keep the shape of the output tensor.
  return a @ ones(a.shape[0])[:, None]


def outer(a: TT[["i"]], b: TT[["j"]]) -> TT[["i", "j"]]:
  return a[:, None] * b


def diag(a: TT[["i", "i"]]) -> TT[["i"]]:
  return a[arange(a.shape[0]), arange(a.shape[0])]


def eye(j: int) -> TT[["j", "j"]]:
  return (arange(j)[:, None] == arange(j)) * 1


def triu(j: int) -> TT[["j", "j"]]:
  return (arange(j)[:, None] <= arange(j)) * 1


def cumsum(a: TT[["i"]]) -> TT[["i"]]:
  return a @ triu(a.shape[0])


def diff(a: TT[["i"]]) -> TT[["i - 1"]]:
  # TODO: tinygrad sometimes return tensors as floats here
  return a[(arange(a.shape[0] - 1) + 1).cast(dtypes.int)] - a[arange(a.shape[0] - 1).cast(dtypes.int)]


def vstack(a: TT[["i"]], b: TT[["i"]]) -> TT[[2, "i"]]:
  return Tensor([[1], [0]]) * a + Tensor([[0], [1]]) * b


# TODO: should I make i as tensor to show in diagrams, or should I show sizes in diagrams directly instead?
def roll(a: TT[["i"]], i: int) -> TT[["i"]]:
  return a[_m((arange(i) + 1), i)]


def flip(a: TT[["i"]], i: int) -> TT[["i"]]:
  return a[:i:][::-1]


def compress(g: TT[["i"], dtypes.bool], v: TT[["i"]], i: int) -> TT[["i"]]:
  return (g * cumsum(1 * g) == (arange(i) + 1)[:, None]) @ v


def pad_to(a: TT[["i"]], i: int, j: int) -> TT[["j"]]:
  raise NotImplementedError


def sequence_mask(values: TT[["i", "j"]], length: TT[["i"], dtypes.int]) -> TT[["i", "j"]]:  # fmt: off
  raise NotImplementedError


def bincount(a: TT[["i"]], j: int) -> TT[["j"]]:
  raise NotImplementedError


def scatter_add(values: TT[["i"]], link: TT[["i"]], j: int) -> TT[["j"]]:
  raise NotImplementedError


def flatten(a: TT[["i", "j"]], i: int, j: int) -> TT[["i * j"]]:
  raise NotImplementedError


def linspace(i: TT[[1]], j: TT[[1]], n: int) -> TT[["n"], dtypes.float]:
  raise NotImplementedError


def heaviside(a: TT[["i"]], b: TT[["i"]]) -> TT[["i"]]:
  raise NotImplementedError


def repeat(a: TT[["i"]], d: TT[[1]]) -> TT[["d", "i"]]:
  raise NotImplementedError


def bucketize(v: TT[["i"]], boundaries: TT[["j"]]) -> TT[["i"]]:
  raise NotImplementedError
