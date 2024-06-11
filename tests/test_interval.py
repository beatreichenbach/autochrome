import logging


logger = logging.getLogger(__name__)


def smoothstep(x: float) -> float:
    return x**2 * (3.0 - 2.0 * x)


def find_interval(values: list[float], x: float) -> int:
    # this is just an algorithm to the index that is closest to x in values.
    left = 0
    last_interval = len(values) - 2
    size = last_interval

    while size > 0:
        # half = int(size / 2)
        half = size >> 1
        middle = left + half + 1
        if values[middle] <= x:
            left = middle
            size -= half + 1
        else:
            size = half
    interval = min(last_interval, left)

    return interval


def test_find_interval():
    resolution = 8
    scale = [smoothstep(smoothstep(k / (resolution - 1))) for k in range(resolution)]
    logger.debug('\n'.join([f'{i}: {x}' for i, x in enumerate(scale)]))
    x = find_interval(scale, 0.1)
    logger.debug(x)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    test_find_interval()
