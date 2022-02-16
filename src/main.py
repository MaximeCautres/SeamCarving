from pickletools import uint8
import numpy as np
import cv2
import os
import datetime
import math
import argparse
from PIL import ImageFilter
from scipy import signal


def colorized_gradient(img):
    """
    A Function used to debug to observe the gradient
    """
    rows, cols = img.shape
    colorized_gradient_img = np.zeros((*img.shape, 3))
    for i in range(rows):
        for j in range(cols):
            if img[i, j] >= 0:
                colorized_gradient_img[i, j, :] = np.array([0, 0, img[i, j]])
            else:
                colorized_gradient_img[i, j, :] = np.array([-img[i, j], 0, 0])
    return (colorized_gradient_img/np.amax(colorized_gradient_img)*255).astype('uint8')


def get_edge_gradient(source):
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


def get_minimal_path_image(gradient):
    """
    This function takes in entry the gradient and computes the
    image of potential energy.
    """
    rows, cols = gradient.shape
    path_values = np.zeros(gradient.shape)
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
        line[i+1] = line[i]-1*(0 != line[i]) + np.argmin(paths[i+1,
                                                               max(0, line[i]-1):min(cols, line[i]+2)])
        paths[i+1, line[i+1]] = 255
    return paths, line


def get_rid_of_line(source, line):
    """
    This function takes in entry the source and the path
    of lowest energy and returns the source were paths pixels
    have been removed.
    """
    rows, cols, deph = source.shape
    new_source = np.zeros((rows, cols-1, deph))
    for i in range(rows):
        new_source[i, :line[i]] = source[i, :line[i]]
        new_source[i, line[i]:] = source[i, line[i]+1:]
    return new_source


def seam_carving_x(source, step):
    """
    The function takes in entry an image source and a parameter
    step specifying the number of step to do in the seam carving
    algorithm. The function returns the image with step less rows.
    """
    print(step, source)
    rang = source.shape[1]//2
    if step is not None:
        rang = step

    for k in range(rang):
        # juste for the style
        if step is not None:
            tty_sizes = os.popen('stty size', 'r').read().split()
            if tty_sizes == []:
                columns = 80  # default
            else:
                _, columns = map(int, tty_sizes)
            message = f"Computing step n°{k} of {step} <>"
            width = columns - len(message)
            advancment = math.ceil(width * (k/step))
            if k != step-1:
                print(
                    f"Computing step n°{k} of {step} <{'#'*advancment}{' '*(width-advancment)}>", end='\r')
            else:
                print(f"Computing step n°{k} of {step} <{'#'*width}>")
        else:
            print(f"Computing step n°{k} of {rang}")
        # end of the flex

        gradient = get_edge_gradient(source)
        paths = get_minimal_path_image(gradient)
        best_path, line = get_minimal_path_image_highlight(paths)
        source = get_rid_of_line(source, line)

        if step is None:
            cv2.imshow("bestpath", best_path.astype('uint8'))
            cv2.waitKey(0)

    # Add security in case of mis the end
    if step is None:
        b = input("Warning do not enter any more, except for validating the access message and after to quit. Acces Message := Yes\n::=")
        while b != 'Yes':
            b = input("Wrong access message, Acces Message := Yes\n::= ")
    return source


def main_seam(step, src_path, output_path, force_write=False):
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
    # TODO Add an automatic save that not erase already existing file

    if not force_write:
        if os.path.exists(output_path):
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H-%M-%S")
            new_name = f"img_{step}_steps_{timestamp}.png"
            output_path = os.path.join(os.path.dirname(output_path), new_name)
            print(
                f"Output path exists.  Use -f to forcibly overwrite.  Will write output to {output_path}.")

    out_dir = os.path.dirname(output_path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Load an color image
    print("The source image is loading", end="\r")
    source = cv2.imread(src_path, cv2.IMREAD_COLOR)
    print("The source image is loaded ")

    # Seam Carving part
    print("Beginning of the seam carving computations")
    step = step if step else None  # Convert 0 to None
    output = seam_carving_x(source, step)
    print("The seam carving computation have finished")

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
    args = parser.parse_args()

    main_seam(args.step[0], args.input, args.output, args.force)
