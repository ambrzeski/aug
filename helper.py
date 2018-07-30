def show_progress(iterable, step):
    count = len(iterable)
    for i, e in enumerate(iterable):
        if i % step == 0:
            print("{}/{}".format(i, count))
        yield e
    if count % step != 0:
        print("{}/{}".format(count, count))


def to_string(x):
    str_parts = []
    for p in x:
        if isinstance(p, float):
            str_parts.append("{:.2f}".format(p))
        else:
            str_parts.append(str(p))
    return "-".join(str_parts)


def is_mask(img):
    return len(img.shape) == 2
