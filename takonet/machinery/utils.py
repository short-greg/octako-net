
def calc_conv_out(in_: int, k: int, stride: int, dilation: int, padding: int) -> float:
    
    return (in_ + 2 * padding - dilation * (k - 1) - 1) / stride + 1


def calc_conv_k(in_: int, out_: int, stride: int, dilation: int, padding: int) -> float:
    return -(stride * (out_ - 1) - in_ - 2 * padding + 1) / dilation + 1


def calc_conv_padding(in_: int, out_: int, k: int, stride: int, dilation: int) -> float:
    return ((out_ - 1) * stride - in_ + dilation * (k - 1) + 1) / 2


def calc_conv_transpose_out(in_: int, k: int, stride: int, padding: int, dilation: int, out_padding: int) -> float:
    return float((in_ - 1) * stride - 2 * padding + dilation * (k - 1) + out_padding) + 1


def calc_conv_transpose_k(in_: int, out_: int, stride: int, padding: int, dilation: int, out_padding: int) -> float:
    return (out_ - stride * (in_ - 1) + 2 * padding - out_padding) / dilation + 1


def calc_max_pool_out(in_: int, k: int, stride: int, dilation: int, padding: int):
    return calc_conv_out(in_, k, stride, dilation, padding)


def calc_pool_k(in_: int, out_: int, stride: int, dilation: int, padding: int) -> float:
    return calc_conv_k(in_, out_, stride, dilation, padding)


def calc_pool_padding(in_: int, out_: int, k: int, stride: int, dilation: int) -> float:
    return calc_conv_padding(in_, out_, k, stride, dilation)


def calc_unpool_out(in_: int, k: int, stride: int, padding: int) -> float:
    return float((in_ - 1) * stride - 2 * padding + k)


def calc_unpool_k(in_: int, out_: int, stride: int, padding: int) -> float:
    return float(-((in_ - 1) * stride - 2 * padding - out_))


def int_to_tuple(val, n_dims):
    if isinstance(val, tuple):
        return val

    return tuple([val] * n_dims)


def calc_conv_out(in_size, ks, strides, paddings):

    dimensions = in_size[2:]
    n = len(dimensions)
        
    ks = int_to_tuple(ks, n)
    strides = int_to_tuple(strides, n)
    paddings = int_to_tuple(paddings, n)
    out_size = list()
    for sz, k, stride, padding in zip(dimensions, ks, strides, paddings):
        if sz == -1:
            out_size.append(-1)
        else:
            out_size.append(calc_conv_out(sz, k, stride, 1, padding))
    
    return tuple(out_size)


def calc_conv_transpose_out(in_size, ks, strides, paddings):

    dimensions = in_size[2:]
    n = len(dimensions)
        
    ks = int_to_tuple(ks, n)
    strides = int_to_tuple(strides, n)
    paddings = int_to_tuple(paddings, n)
    out_size = list()
    for sz, k, stride, padding in zip(dimensions, ks, strides, paddings):
        if sz == -1:
            out_size.append(-1)
        else:
            out_size.append(calc_conv_transpose_out(sz, k, stride, 1, padding))
    
    return tuple(out_size)


def calc_pool_out(in_size, ks, strides, paddings):

    dimensions = in_size[2:]
    n = len(dimensions)
        
    ks = int_to_tuple(ks, n)
    strides = int_to_tuple(strides, n)
    paddings = int_to_tuple(paddings, n)
    out_size = list()
    for sz, k, stride, padding in zip(dimensions, ks, strides, paddings):
        if sz == -1:
            out_size.append(-1)
        else:
            out_size.append(calc_pool_out(sz, k, stride, 1, padding))
    
    return tuple(out_size)


def calc_maxunpool_out(in_size, ks, strides, paddings):

    dimensions = in_size[2:]
    n = len(dimensions)
        
    ks = int_to_tuple(ks, n)
    strides = int_to_tuple(strides, n)
    paddings = int_to_tuple(paddings, n)
    out_size = list()
    for sz, k, stride, padding in zip(dimensions, ks, strides, paddings):
        if sz == -1:
            out_size.append(-1)
        else:
            out_size.append(calc_unpool_out(sz, k, stride, padding))
    
    return tuple(out_size)

def to_int(size):
    return map(int, size)
