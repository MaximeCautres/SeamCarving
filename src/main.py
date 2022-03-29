from pickletools import uint8
from xmlrpc.client import Boolean
from cv2 import seamlessClone
import numpy as np
import cv2
import os
import datetime
import math
import argparse
from PIL import ImageFilter
from scipy import signal


def run_app(path):
    print(f"Running seam carving on {path}")


def get_rows_cols(img, cut_horizontal: Boolean):
    if cut_horizontal:
        row, col = img.shape
        return col, row
    else:
        return img.shape


# def colorized_gradient(img, cut_horizontal: Boolean):
#     """
#     A Function used to debug to observe the gradient
#     """
#     rows, cols = get_rows_cols(img, cut_horizontal)
#     colorized_gradient_img = np.zeros((*img.shape, 3))
#     for i in range(rows):
#         for j in range(cols):
#             if img[i, j] >= 0:
#                 colorized_gradient_img[i, j, :] = np.array([0, 0, img[i, j]])
#             else:
#                 colorized_gradient_img[i, j, :] = np.array([-img[i, j], 0, 0])
#     return (colorized_gradient_img/np.amax(colorized_gradient_img)*255).astype('uint8')


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


def get_minimal_path_image(gradient_img: np.ndarray, cut_horizontal: Boolean):
    """
    This function takes in entry the gradient and computes the
    image of potential energy.
    """
    # if we're cutting horizontally, we need to transpose the gradient
    # map so that indexing is corrent
    if cut_horizontal:
        gradient = np.transpose(gradient_img)
    else:
        gradient = gradient_img
    rows, cols = gradient.shape
    path_values = np.zeros((rows, cols))
    for i in range(rows-2, -1, -1):
        for j in range(cols):
            path_values[i, j] = gradient[i, j] + min(path_values[i+1, max(0, j-1)], min(
                path_values[i+1, j], path_values[i+1, min(j+1, cols-1)]))
    return path_values/np.max(path_values)*255


def get_minimal_path_image_highlight(paths):
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


def get_rid_of_line(source, line, cut_horizontal):
    """
    This function takes in entry the source and the path
    of lowest energy and returns the source were paths pixels
    have been removed.
    """
    if cut_horizontal:
        # transpose messes with colour axis too
        tmp = np.swapaxes(source, 0, 1)
    else:
        tmp = source
    rows, cols, nb_colours = tmp.shape
    new_source = np.zeros((rows, cols-1, nb_colours))
    for i in range(rows):
        new_source[i, :line[i]] = tmp[i, :line[i]]
        new_source[i, line[i]:] = tmp[i, line[i]+1:]
    if cut_horizontal:
        new_source = np.swapaxes(new_source, 0, 1)
    return new_source


def write_progress_to_console(step, nb_steps):
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


def seam_carving_x(source: np.ndarray, steps: int, cut_horizontal: Boolean):
    """
    The function takes in entry an image source and a parameter
    steps specifying the number of steps to do in the seam carving
    algorithm. The function returns the image with steps less rows.
    """
    for k in range(steps):

        # write_progress_to_console(k, steps)

        # O(n^2 * d^2)
        gradient_map = get_edge_gradient(source)
        paths = get_minimal_path_image(gradient_map, cut_horizontal)
        best_path, line = get_minimal_path_image_highlight(paths)
        source = get_rid_of_line(source, line, cut_horizontal)

        if steps is None:
            cv2.imshow("bestpath", best_path.astype('uint8'))
            cv2.waitKey(0)

    return source


def main_seam(step: int, src_path: str, output_path: str, force_write: Boolean, cut_horizontal: Boolean):
    """
    This function is an encapsulation for the seam
    carving. The function manages the argv entry
    to get the correct input and output, while
    managing the end print.
    """
    if not os.path.exists(src_path):
        print("Source file not found.  Exiting...")
        return
    # Allow the choice for the save path directly from the call
    # We do it now in order to have a early fail

    if not force_write:
        if os.path.exists(output_path):
            old = output_path
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
            new_name = f"img_{step}_steps_{timestamp}.png"
            output_path = os.path.join(os.path.dirname(old), new_name)
            print(
                f"File {old} already exists.  Refusing to overwrite (use option -f to overwrite).  Will write output to {output_path}.")

    out_dir = os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load an color image
    print("The source image is loading", end="\r")
    source = cv2.imread(src_path, cv2.IMREAD_COLOR)
    print("The source image is loaded ")

    # Seam Carving part
    output = seam_carving_x(source, step, cut_horizontal)

    # Save the output
    cv2.imwrite(output_path, output)
    print(f"Output written to {output_path}")

    # Display the final result
    cv2.imshow('source', source.astype('uint8'))
    cv2.imshow('output', output.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply the seam carving methode to an input image.')
    parser.add_argument('step', nargs=1, type=int,
                        help="Specify the number of columns to remove in the seam carving")
    parser.add_argument('-i', '--input', dest="input", type=str,
                        default="../images/sunset.png",
                        help='Specify the path to the input image, default correspond to an example')
    parser.add_argument('-o', '--output', dest="output", type=str,
                        default="./output/im1.png",
                        help='Specify a path to write the output image, default correspond to an example')
    parser.add_argument('-f', '--force', dest="force", action='store_true',
                        help="Force writing of output file, even if it overwrite an existing file")
    parser.add_argument('--horizontal', dest="horizontal", action='store_true')
    args = parser.parse_args()

    main_seam(args.step[0], args.input, args.output,
              args.force, args.horizontal)
