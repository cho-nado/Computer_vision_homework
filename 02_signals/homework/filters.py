import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """

    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))
    half_width = Wk // 2
    half_height = Hk // 2

    for x in range(Wi):
        for y in range(Hi):
            filtered_value = 0
            for fx in range(Wk):
                for fy in range(Hk):
                    img_x = x + (half_width - fx)
                    img_y = y + (half_height - fy)

                    pixel_value = image[img_y, img_x] if 0 <= img_x < Wi and 0 <= img_y < Hi else 0

                    filtered_value += pixel_value * kernel[fy, fx]
            out[y, x] = filtered_value

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = np.zeros((H + 2* pad_height, W + 2 * pad_width))
    out[pad_height : pad_height + H, pad_width : pad_width + W] = image
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # Изменяем порядок элементов ядра
    flipped_kernel = np.flip(kernel)

    # Добавляем нулевые границы к изображению
    padded_image = zero_pad(image, Hk // 2, Wk // 2)

    # Проходим по каждому пикселю изображения
    for x in range(Wi):
        for y in range(Hi):
            out[y, x] = (flipped_kernel * padded_image[y: y+ Hk , x: x + Wk]).sum()

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros_like(f)

    for row in range(Hf - Hg + 1):
        for col in range(Wf - Wg + 1):
            fragment = f[row:row + Hg, col:col + Wg]
            fragment_average = np.mean(fragment)

            elementwise_product = g * (fragment - fragment_average)

            out[row, col] = np.sum(elementwise_product)

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    Hf, Wf = f.shape
    Hg, Wg = g.shape

    centered_g = g - np.mean(g)

    # Проходим по всем возможным позициям в f
    for row in range(Hf - Hg + 1):
        for col in range(Wf - Wg + 1):
            fragment = f[row:row + Hg, col:col + Wg]

            product = centered_g * (fragment - np.mean(fragment))

            out[row, col] = np.sum(product)

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = np.zeros_like(f)
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    g_centered = g - np.mean(g)
    g_std_dev = np.std(g_centered)

    for i in range(Hf - Hg + 1):
        for j in range(Wf - Wg + 1):
            fragment = f[i:i + Hg, j:j + Wg]
            normalized_product = np.multiply(g_centered, fragment - np.mean(fragment))
            out[i, j] = np.sum(normalized_product) / (np.std(fragment) * g_std_dev)

    return out
