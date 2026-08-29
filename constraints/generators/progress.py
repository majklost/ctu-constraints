"""Optional progress display for long-running generator loops."""

from collections.abc import Iterable, Iterator

from tqdm import tqdm


def track[T](
    iterable: Iterable[T],
    *,
    enabled: bool,
    description: str,
    total: int | None = None,
    unit: str = "sample",
) -> Iterator[T]:
    """Iterate normally or through a consistently configured tqdm bar."""
    if not enabled:
        yield from iterable
        return
    yield from tqdm(
        iterable,
        desc=description,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        mininterval=1.0,
    )
