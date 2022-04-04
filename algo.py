from hashlib import new
from pickletools import uint8
from unicodedata import decimal
import random
from unittest import case
import cv2
import numpy as np
from PIL import Image
import os
import datetime
import math
import argparse
from scipy import signal
import typing


class ParsingError(Exception):
    pass


def get_rows_cols(img, cut_horizontal):
    if cut_horizontal:
        row, col = img.shape
        return col, row
    else:
        return img.shape


def get_edge_gradient(source: np.ndarray):
    """
    This function takes an image and returns its gradient.
    The gradient is the square root of the sum of the square of
    the gradient on x and y.
    """
    # first step get the gray scale
    grey = (0.2125*source[:, :, 0]+0.7154*source[:, :,
            1]+.0721*source[:, :, 2]).astype('uint8')
    # Compute the kernel used for the edge recognition
    # print(Kernel.sobel())
    kernel_y = np.array([[-.125, -.25, -.125], [0, 0, 0], [.125, .25, .125]])
    kernel_x = np.array([[-.125, -.25, -.125], [0, 0, 0],
                        [.125, .25, .125]]).transpose()
    # apply the kernel on the image
    edge_y = signal.convolve2d(grey, kernel_y, mode='full', boundary='symm')[
        1:-1, 1:-1].astype('int16')
    edge_x = signal.convolve2d(grey, kernel_x, mode='full', boundary='symm')[
        1:-1, 1:-1].astype('int16')
    # add both gradient in order to get the edge
    result = (np.abs(edge_x)**2 + np.abs(edge_y)**2)**(1/2)
    result = result/np.max(result)*255
    return result


def get_minimal_path_image(gradient: np.ndarray):
    """
    This function takes in entry the gradient and computes the
    image of potential energy.
    """
    # if we're cutting horizontally, we need to transpose the gradient
    # map so that indexing is corrent
    rows, cols = gradient.shape
    path_values = np.zeros((rows, cols))
    for i in range(rows-2, -1, -1):
        for j in range(cols):
            path_values[i, j] = gradient[i, j] + min(path_values[i+1, max(0, j-1)],
                                                     min(path_values[i+1, j], path_values[i+1, min(j+1, cols-1)]))
    return path_values/np.max(path_values)*255


def get_minimal_path_image_highlight(paths: np.ndarray):
    """
    This function tsjke in entry a map of potential energy
    and returns the lowest energy path in line and draw it
    in paths
    """
    rows, cols = paths.shape
    line = np.zeros(rows).astype('uint32')
    line[0] = np.argmin(paths[0])
    paths[0, line[0]] = 255
    for i in range(rows-1):
        # python slicing [min:max] is inclusive of min but exclusive of max
        # argmin is taking a value a 0,1,2.  Subtract the one to rescale provided
        # we're not on the first column of the array
        line[i+1] = line[i]-1*(0 != line[i]) + np.argmin(paths[i+1,
                                                               max(0, line[i]-1):min(cols, line[i]+2)])
        paths[i+1, line[i+1]] = 255
    return paths, line


def get_rid_of_line(source: np.ndarray, paths: np.ndarray, line):
    """
    This function takes in entry the source and the path
    of lowest energy and returns the source were paths pixels
    have been removed.
    """
    rows, cols, nb_colours = source.shape
    new_source = np.zeros((rows, cols-1, nb_colours))
    new_paths = np.zeros((rows, cols-1))
    for i in range(rows):
        new_source[i, :line[i]] = source[i, :line[i]]
        new_source[i, line[i]:] = source[i, line[i]+1:]
        new_paths[i, :line[i]] = paths[i, :line[i]]
        new_paths[i, line[i]:] = paths[i, line[i]+1:]
    return new_source, new_paths


def write_progress_to_console(step: int, nb_steps: int):
    """
    This function takes in entry a counter and its maximum
    and manage a nice adapt to terminal prompt for the progression
    of the seam carving algorithm
    """
    if nb_steps is not None:
        tty_sizes = os.popen('stty size', 'r').read().split()
        if tty_sizes == []:
            columns = 80  # default
        else:
            _, columns = map(int, tty_sizes)
        message = f"Computing nb_steps n°{step} of {nb_steps} <>"
        width = columns - len(message)
        advancment = math.ceil(width * (step/nb_steps))
        if step != nb_steps-1:
            print(
                f"Computing step n°{step} of {nb_steps} <{'#'*advancment}{' '*(width-advancment)}>", end='\r')
        else:
            print(f"Computing step n°{step} of {nb_steps} <{'#'*width}>")
    else:
        print(f"Computing step n°{step} of {nb_steps}")


def seam_carving_x(source: np.ndarray, steps: int, recompute_every: int,
                   print_result: bool = True):
    """
    The function takes in entry an image source and a parameter
    steps specifying the number of steps to do in the seam carving
    algorithm. The function returns the image with steps less rows.
    """
    for k in range(steps):
        if print_result:
            write_progress_to_console(k, steps)

        # O(n^2 * d^2)
        gradient_map = get_edge_gradient(source)
        paths = get_minimal_path_image(gradient_map)
        for _ in range(recompute_every):
            best_path, line = get_minimal_path_image_highlight(paths)
            source, paths = get_rid_of_line(
                source, best_path, line)

    return source


def manage_input(input_string: str, input_size: int, dim_name: str) -> int:
    """
    The function as for goal to check the validity of the args give
    to --height or --width argument when the main.Py is called in the
    console. Moreover, the function also return a normalized version
    of the input_string which does not depend on what type of data is in entry.
    """
    if input_string[-1] == "%":
        try:
            input_string = int(input_string[:-1])
        except ValueError as e:
            raise ParsingError(
                f"The {dim_name} in ratio {e.split(' ')[-1]} is not of the form 'integer%'.")
        if not 0 <= input_string <= 100:
            raise ParsingError(
                f"The {dim_name} in ratio {str(input_string)} is a percentage and should be between 0 and 100."
            )
    else:
        try:
            input_string = int(input_string)
        except ValueError as e:
            raise ParsingError(
                f"The {dim_name} in pixel of the output {e.split(' ')[-1]} is not of the form 'integer'.")
        if not 0 <= input_string <= input_size:
            raise ParsingError(
                f"The {dim_name} in pixel of the output {str(input_string)} should be between 0 and {str(input_size)}."
            )
        input_string = int(100*input_string/input_size)
    return input_string


def get_rescaled_version(source: np.ndarray, width: int, height: int, cols: int,
                         rows: int, recompute_every: int):
    """
    This function compute a seam carving of a pictures were both dimension
    need to be reduced. However here, the specificity is that we start by
    doing a rescale in order to have only one dimension to seam carve after,
    this reduce the possibility of creation of artefact as is minimize the
    paths to remove from the pictures
    """
    # get the less_affected dimension
    cut_vertical = np.argmax(np.array((width, height)))
    resize_ratio = height if cut_vertical else width
    # dimension of the source image after the rescale
    new_cols = int(cols * resize_ratio / 100)
    new_rows = int(rows * resize_ratio / 100)
    # new_source is the rescaled source
    new_source = cv2.resize(source, (new_cols, new_rows),
                            interpolation=cv2.INTER_CUBIC)
    # compute the original wanted picture
    out_cols = int(cols*width/100)
    out_rows = int(rows*height/100)

    # compute the number of path to remove for the seamcarving part
    step = max(new_cols-out_cols, new_rows-out_rows)
    big_step = step//recompute_every
    little_step = step % recompute_every
    # Seam Carving part
    if not cut_vertical:
        output = np.swapaxes(seam_carving_x(np.swapaxes(new_source, 0, 1), big_step,
                                            recompute_every), 0, 1)
        output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), 1,
                                            little_step), 0, 1)
    else:
        output = seam_carving_x(new_source, big_step, recompute_every)
        output = seam_carving_x(output, 1, little_step)
    return output, step


def carv_no_rescale_balanced(output: np.ndarray, big_cols_step: int, big_rows_step: int,
                             little_cols_step: int, little_rows_step: int,
                             recompute_every: int):
    """
    This function compute a seam carving of a pictures were both dimension
    need to be reduced. However here, we are doing the seam carving on both
    dimension and more prescisly we try to alternate proportionally between
    each dimension according to the number of path to remove on each dimension.
    """
    prop_factor = (big_cols_step + 1)/(big_rows_step+1)
    prop_factor = max(prop_factor, 1/prop_factor)
    abscisse_size = min(big_cols_step, big_rows_step)+1
    how_many_to_remove = []
    for k in range(1, abscisse_size+1):
        how_many_to_remove.append(
            int(prop_factor*k)-int((k-1)*prop_factor))
    if big_cols_step < big_rows_step:
        rows_step_list = how_many_to_remove
        cols_step_list = [1]*abscisse_size
    else:
        cols_step_list = how_many_to_remove
        rows_step_list = [1]*abscisse_size
    for k in range(abscisse_size):
        write_progress_to_console(k, abscisse_size)
        if k != abscisse_size-1:
            output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), rows_step_list[k],
                                                recompute_every, False), 0, 1)
            output = seam_carving_x(
                output, cols_step_list[k], recompute_every, False)
        else:
            output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), rows_step_list[k]-1,
                                                recompute_every, False), 0, 1)
            output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), 1,
                                                little_rows_step, False), 0, 1)
            output = seam_carving_x(
                output, cols_step_list[k]-1, recompute_every, False)
            output = seam_carving_x(output, 1, little_cols_step, False)
    return output


def carv_no_rescale_random(output: np.ndarray, big_cols_step: int, big_rows_step: int,
                           little_cols_step: int, little_rows_step: int, recompute_every: int,
                           step: int):
    """
    This function compute a seam carving of a pictures were both dimension
    need to be reduced. However here, we are doing the seam carving on both
    dimension and more prescisly we chose at each step which dimension is
    carved using a Bernouilly random variable of parameter the proportion
    of path to remove on the rows on all path to remove.
    """
    k = 0
    count_rows = 0
    count_cols = 0
    while k < step:
        write_progress_to_console(k, step)
        factor = (big_cols_step + 1)/step
        col_or_row = random.random() >= factor
        count = 1
        if col_or_row:
            while (random.random() >= factor) == col_or_row and count_rows + count < big_rows_step+1:
                count += 1
            count_rows += count
            if count_rows <= big_rows_step:
                output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), count,
                                                    recompute_every, False), 0, 1)
            else:
                output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), count-1,
                                                    recompute_every, False), 0, 1)
                output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), 1,
                                                    little_rows_step, False), 0, 1)
                output = seam_carving_x(output, big_cols_step-count_cols,
                                        recompute_every, False)
                output = seam_carving_x(output, 1,
                                        little_cols_step, False)
                k = step
        else:
            while (random.random() >= factor) == col_or_row and count_cols + count < big_cols_step+1:
                count += 1
            count_cols += count
            if count_cols <= big_cols_step:
                output = seam_carving_x(
                    output, count, recompute_every, False)
            else:
                output = seam_carving_x(output, count-1,
                                        recompute_every, False)
                output = seam_carving_x(output, 1,
                                        little_cols_step, False)
                output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), big_rows_step-count_rows,
                                                    recompute_every, False), 0, 1)
                output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), 1,
                                                    little_rows_step, False), 0, 1)
                k = step
        k += count
    return output


def carv_no_rescale_one_before_the_other(output: np.ndarray, cols_step: int, rows_step: int,
                                         big_cols_step: int, big_rows_step: int, little_cols_step: int,
                                         little_rows_step: int, recompute_every: int):
    """
    This function compute a seam carving of a pictures were both dimension
    need to be reduced. However here, we are doing the seam carving on both
    dimension and more prescisly it start by removing all the paths to remove
    on the cols and it finally removes all the paths on the rows.
    """
    write_progress_to_console(0, cols_step+rows_step)
    output = seam_carving_x(output, big_cols_step,
                            recompute_every, False)
    write_progress_to_console(
        big_cols_step*recompute_every, cols_step+rows_step)
    output = seam_carving_x(output, 1,
                            little_cols_step, False)
    write_progress_to_console(
        big_cols_step*recompute_every + little_cols_step, cols_step+rows_step)
    output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), big_rows_step,
                                        recompute_every, False), 0, 1)
    write_progress_to_console(big_cols_step*recompute_every + little_cols_step +
                              big_rows_step*recompute_every, cols_step+rows_step)
    output = np.swapaxes(seam_carving_x(np.swapaxes(output, 0, 1), 1,
                                        little_rows_step, False), 0, 1)
    write_progress_to_console(big_cols_step*recompute_every + little_cols_step +
                              big_rows_step*recompute_every+little_rows_step, cols_step+rows_step)
    return output


def get_no_rescaled_version(source: np.ndarray, width: int, height: int, cols: int, rows:
                            int, recompute_every: int, carving_methode: str):
    """
    This function compute a seam carving of a pictures were both dimension
    need to be reduced. However here, we are doing the seam carving on both
    dimension. There is multiple ways to do it, you can do it by alternating,
    choosing randomly, are by doing dimension one after the other. This function
    compute common data used by the three variant above and call the good one.
    """
    # get the new image size
    new_cols = int(cols * width / 100)
    new_rows = int(rows * height / 100)
    # compute the path to carv on each dim
    cols_step = cols-new_cols
    rows_step = rows-new_rows
    # compute the way this path will be removed
    big_cols_step = cols_step//recompute_every
    little_cols_step = cols_step % recompute_every
    big_rows_step = rows_step//recompute_every
    little_rows_step = rows_step % recompute_every
    step = big_cols_step + big_rows_step + 2
    # remove them as it is required by the user
    output = source.copy()
    if carving_methode == 'balanced':
        output = carv_no_rescale_balanced(
            output, big_cols_step, big_rows_step, little_cols_step,
            little_rows_step, recompute_every)
    elif carving_methode == 'random':
        output = carv_no_rescale_random(output, big_cols_step, big_rows_step,
                                        little_cols_step, little_rows_step, recompute_every, step)
    elif carving_methode == 'width->height':
        output = carv_no_rescale_one_before_the_other(
            output, cols_step, rows_step, big_cols_step, big_rows_step,
            little_cols_step, little_rows_step, recompute_every)
    else:
        output = np.swapaxes(carv_no_rescale_one_before_the_other(
            np.swapaxes(
                output, 0, 1), rows_step, cols_step, big_rows_step, big_cols_step,
            little_rows_step, little_cols_step, recompute_every), 0, 1)
    return output, step


def main_seam(recompute_every: int, src_path: str, output_path: str, force_write: bool,
              height: int, width: int, norescale: bool, carving_methode: str):
    """
    This function is an encapsulation for the seam
    carving. The function manages the argv entry
    to get the correct input and output, while
    managing the end print.
    """
    if not os.path.exists(src_path):
        print("Source file not found.  Exiting...")
        return

    out_dir = os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load an color image
    print("The source image is loading", end="\r")
    # img_src = Image.open(src_path)
    # src_4channel = np.asarray(img_src)
    # source = src_4channel[..., :3]  # drop the transparency channel
    source = cv2.imread(src_path, cv2.IMREAD_COLOR)
    rows, cols, _ = source.shape
    print("The source image is loaded ")

    # Manage the input
    width = manage_input(width, cols, 'width')
    height = manage_input(height, rows, 'height')
    if width == 100 or height == 100:
        norescale = False

    if norescale:
        output, step = get_no_rescaled_version(
            source, width, height, cols, rows, recompute_every, carving_methode)
    else:
        output, step = get_rescaled_version(
            source, width, height, cols, rows, recompute_every)

    if not force_write:
        if os.path.exists(output_path):
            old = output_path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
            new_name = f"img_{step}_steps_{timestamp}.png"
            output_path = os.path.join(os.path.dirname(old), new_name)
            print(
                f"File {old} already exists.  Refusing to overwrite (use option -f to overwrite).  Will write output to {output_path}.")

    # Save the output
    cv2.imwrite(output_path, output)
    print(f"Output written to {output_path}")

    return output_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply the seam carving methode to an input image.')
    parser.add_argument('--height', dest="height", type=str, default="100%",
                        help="specify the percentage of the height of the input image you want to reach for the height for the output image")
    parser.add_argument('--width', dest="width", type=str, default="100%",
                        help="specify the percentage of the width of the input image you want to reach for the width for the output image")
    parser.add_argument('--per-step', dest="per_step", type=int,
                        default=1,
                        help='Specify how many path to remove at each step.')
    parser.add_argument('-i', '--input', dest="input", type=str,
                        default="./images/sunset.png",
                        help='Specify the path to the input image, default correspond to an example')
    parser.add_argument('-o', '--output', dest="output", type=str,
                        default="./output/im1.png",
                        help='Specify a path to write the output image, default correspond to an example')
    parser.add_argument('-f', '--force', dest="force", action='store_true',
                        help="Force writing of output file, even if it overwrite an existing file")
    parser.add_argument("--norescale", dest="norescale", action="store_true", default=False,
                        help="Specify if you want to use the scaling when you carv in 2 dimension or only using seam carving.")
    parser.add_argument("--2D-carving-method", dest="carving_methode",
                        type=str, choices=["height->width",
                                           "width->height", "random", "balanced"],
                        default="balanced",
                        help="Specify the way seam carving of both dimension a perform:\n ** 'height->width':" +
                        "start by the seam carving on the height and after on the width. \n ** 'width->height':" +
                        "start by the seam carving on the width and after on the height. \n ** 'random': chose" +
                        "randomly using a boolean law (well balanced) on which dimension the seam carving" +
                        "will be applied.\n ** 'balanced': the seam carving on each dimension are uniformly distributed threw time.")
    args = parser.parse_args()

    try:
        main_seam(args.per_step, args.input, args.output,
                  args.force, args.height, args.width, args.norescale, args.carving_methode)
    except ParsingError as error:
        print(f"ParsingError: {error}")
