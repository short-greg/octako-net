
def calc_conv_out(in_: int, k: int, stride: int, dilation: int, padding: int) -> float:
    
    return (in_ + 2 * padding - dilation * (k - 1) - 1) / stride + 1


def calc_conv_k(in_: int, out_: int, stride: int, dilation: int, padding: int) -> float:
    return -(stride * (out_ - 1) - in_ - 2 * padding + 1) / dilation + 1


def calc_conv_transpose_out(in_: int, k: int, stride: int, padding: int, dilation: int, out_padding: int) -> float:
    return float((in_ - 1) * stride - 2 * padding + dilation * (k - 1) + out_padding) + 1


def calc_conv_transpose_k(in_: int, out_: int, stride: int, padding: int, dilation: int, out_padding: int) -> float:
    return (out_ - stride * (in_ - 1) + 2 * padding - out_padding) / dilation + 1


def calc_max_pool_out(in_: int, k: int, stride: int, dilation: int, padding: int):
    return calc_conv_out(in_, k, stride, dilation, padding)


def calc_max_pool_k(in_: int, out_: int, stride: int, dilation: int, padding: int) -> float:
    return calc_conv_k(in_, out_, stride, dilation, padding)


def calc_max_unpool_out(in_: int, k: int, stride: int, padding: int) -> float:
    return float((in_ - 1) * stride - 2 * padding + k)


def calc_max_unpool_k(in_: int, out_: int, stride: int, padding: int) -> float:
    return float(-((in_ - 1) * stride - 2 * padding - out_))
