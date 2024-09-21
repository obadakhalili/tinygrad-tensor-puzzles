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
  return a[1:] - a[:-1]


def vstack(a: TT[["i"]], b: TT[["i"]]) -> TT[[2, "i"]]:
  return Tensor([[1], [0]]) * a + Tensor([[0], [1]]) * b


# TODO: should I make i as tensor to show in diagrams? should I show sizes in diagrams directly instead?
def roll(a: TT[["i"]], i: int) -> TT[["i"]]:
  return a[_m((arange(i) + 1), i)]


def flip(a: TT[["i"]], i: int) -> TT[["i"]]:
  return a[:i:][::-1]


def compress(g: TT[["i"], dtypes.bool], v: TT[["i"]], i: int) -> TT[["i"]]:
  return (g * cumsum(1 * g) == (arange(i) + 1)[:, None]) @ v


def pad_to(a: TT[["i"]], j: int) -> TT[["j"]]:
  return a @ (arange(a.shape[0])[:, None] == arange(j))


def sequence_mask(values: TT[["i", "j"]], length: TT[["i"], dtypes.int]) -> TT[["i", "j"]]:  # fmt: off
  return (arange(values.shape[1]) < length[:, None]) * values


def bincount(a: TT[["i"]], j: int) -> TT[["j"]]:
  return ones(a.shape[0]) @ (a[:, None] == arange(j))


def scatter_add(value: TT[["i"]], index: TT[["i"]], j: int) -> TT[["j"]]:
  return value @ (index[:, None] == arange(j))


def flatten(a: TT[["i", "j"]]) -> TT[["i * j"]]:
  return a[_fd(arange(p := a.shape[0] * a.shape[1]), a.shape[1]), _m(arange(p), a.shape[1])]  # fmt: off


def linspace(i: TT[[1]], j: TT[[1]], n: int) -> TT[["n"], dtypes.float]:
  return i + (j - i) * (1.0 * arange(n)) / max(1, n - 1)

  # TODO: make this work with input: [0], [0], and 1
  # return ones(n) * i + (cumsum(ones(n) * 0 + (step := (j - i) / (n - 1))) - step)


def heaviside(a: TT[["i"]], b: TT[["i"]]) -> TT[["i"]]:
  return (a > 0) + (a == 0) * b


def repeat(a: TT[["i"]], d: TT[[1]]) -> TT[["d", "i"]]:
  return a * ones(d[0].numpy())[:, None]


def bucketize(v: TT[["i"]], boundaries: TT[["j"]]) -> TT[["i"]]:
  return (v[:, None] >= boundaries) @ ones(boundaries.shape[0])
