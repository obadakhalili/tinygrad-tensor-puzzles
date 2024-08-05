from lib import TinyType as TT
from tinygrad import Tensor, dtypes


def arange(i: int):
  "Use this function to replace a for-loop."
  return Tensor(list(range(i)))


def where(q, a, b):
  "Use this function to replace an if-statement."
  return q * a + q.logical_not() * b


def ones(i: int) -> TT[["i"]]:
  raise NotImplementedError


def sum(a: TT[["i"]]) -> TT[[1]]:
  raise NotImplementedError


def outer(a: TT[["i"]], b: TT[["j"]]) -> TT[["i", "j"]]:
  raise NotImplementedError


def diag(a: TT[["i", "i"]]) -> TT[["i"]]:
  raise NotImplementedError


def eye(j: int) -> TT[["j", "j"]]:
  raise NotImplementedError


def triu(j: int) -> TT[["j", "j"]]:
  raise NotImplementedError


def cumsum(a: TT[["i"]]) -> TT[["i"]]:
  raise NotImplementedError


def diff(a: TT[["i"]], i: int) -> TT[["i"]]:
  raise NotImplementedError


def vstack(a: TT[["i"]], b: TT[["i"]]) -> TT[[2, "i"]]:
  raise NotImplementedError


def roll(a: TT[["i"]], i: int) -> TT[["i"]]:
  raise NotImplementedError


def flip(a: TT[["i"]], i: int) -> TT[["i"]]:
  raise NotImplementedError


def compress(g: TT[["i"], dtypes.bool], v: TT[["i"]], i: int) -> TT[["i"]]:
  raise NotImplementedError


def pad_to(a: TT[["i"]], i: int, j: int) -> TT[["j"]]:
  raise NotImplementedError


def sequence_mask(
  values: TT[["i", "j"]], length: TT[["i"], dtypes.int]
) -> TT[["i", "j"]]:
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
