from pickletools import uint8
import numpy as np
import cv2
import os
import math
import argparse
from scipy import signal

def colorized_gradient(img):
    """
    A Function used to debug to observe the gradient
    """
    rows, cols= img.shape
    colorized_gradient_img = np.zeros((*img.shape, 3))
    for i in range(rows):
        for j in range(cols):
            if img[i, j] >= 0:
                colorized_gradient_img[i, j, :] = np.array([0, 0, img[i, j]])
            else :
                colorized_gradient_img[i, j, :] = np.array([-img[i, j], 0, 0])
    return (colorized_gradient_img/np.amax(colorized_gradient_img)*255).astype('uint8')

def get_edge_gradient(source):
    """
    This function takse in entry an image and returns its grandiant.
    The grandiant is the square root of the sum of the square of 
    the gradiant on x and y.
    """
    # first step get the gray scale
    grey= (0.2125*source[:,:,0]+0.7154*source[:,:,1]+.0721*source[:,:,2]).astype('uint8')
    # Compute the kernel used for the edge recognition
    kernel_y = np.array([[-.125, -.25, -.125], [0, 0, 0], [.125, .25, .125]])
    kernel_x = np.array([[-.125, -.25, -.125], [0, 0, 0], [.125, .25, .125]]).transpose()
    # apply the kernel on the image
    edge_y = signal.convolve2d(grey, kernel_y, mode='full',boundary='symm')[1:-1,1:-1].astype('int16')
    edge_x = signal.convolve2d(grey, kernel_x, mode='full',boundary='symm')[1:-1,1:-1].astype('int16')
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
            path_values[i, j] = gradient[i, j]+ min(path_values[i+1, max(0, j-1)],min(path_values[i+1,j], path_values[i+1, min(j+1, cols-1)]))
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
        line[i+1] = line[i]-1*(0!=line[i]) + np.argmin(paths[i+1, max(0, line[i]-1):min(cols, line[i]+2)])
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
    rang = source.shape[1]//2
    if step is not None:
        rang = step

    for k in range(rang):
        # juste for the style
        if step is not None:
            tty_sizes = os.popen('stty size', 'r').read().split()
            if tty_sizes == []:
                columns = 80 # default
            else:
                _, columns = map(int, tty_sizes)
            message = f"Computing step n°{k} over {step} <>"
            width = columns - len(message)
            advancment = math.ceil(width * (k/step))
            if k != step-1:
                print(f"Computing step n°{k} over {step} <{'#'*advancment}{' '*(width-advancment)}>", end='\r')
            else:
                print(f"Computing step n°{k} over {step} <{'#'*width}>")
        else:
            print(f"Computing step n°{k} over {rang}")
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


def main_seam(input_path, output_path, step):
    """
    This function is an encapsulation for the seam 
    carving. The function manages the argv entry
    to get the correct input and output, while 
    managing the end print.
    """

    IMG_DIR = "../images/"
    IMG_NAME = "sunset.png"
    # Allow the choice of a new image directly from the call
    if input_path is not None:
        input_path = input_path.split('/')
        IMG_NAME = input_path[-1]
        IMG_DIR = "/".join(input_path[:-1])+"/"

    # Allow the choice for the save path directly from the call
    # We do it now in order to have a early fail
    OUT_DIR = "./output/"
    OUT_NAME = "im1.png" # TODO Add an automatic save that not erase already existing file
    if output_path is not None:
        output_path = ("./"+output_path).split('/')
        OUT_NAME = output_path[-1]
        OUT_DIR = "/".join(output_path[:-1])+"/"
        print(OUT_DIR, OUT_NAME)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    file = os.path.join(IMG_DIR, f"{IMG_NAME}")
    if not os.path.exists(file):
        print("File not found.  Exiting...")
        return

    # Load an color image
    print("The source image is loading", end="\r")
    source = cv2.imread(file, cv2.IMREAD_COLOR)
    print("The source image is loaded ")

     # Seam Carving part
    print("Beginning of the seam carving computations")
    output = seam_carving_x(source,step)
    print("The seam carving computation have finished")

    # Display the final result
    cv2.imshow('source', source.astype('uint8'))
    cv2.imshow('output', output.astype('uint8'))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the output
    print(f"Saving the output at {OUT_DIR+OUT_NAME}")
    cv2.imwrite(OUT_DIR+OUT_NAME, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Apply the seam carving methode to an input image.')
    parser.add_argument('-i','--input',dest = "input", type=str, 
        default=None,
        help='Specify the path to the input image, default correspond to an example')
    parser.add_argument('-o','--output',dest = "output", type=str, 
        default=None,
        help='Specify the path to the input image, default correspond to an example')
    parser.add_argument('-s', '--step', dest = "step", type=int,
        default = None,
        help = "Specifie the number of step of the seam carving")
    args = parser.parse_args()

    main_seam(args.input, args.output, args.step)
