import numpy as np
import cv2
import os
import sys
import argparse



def main_seam(input_path, output_path):

    IMG_DIR = "../images/"
    IMG_NAME = "sunset.png"
    # Allow the choice of a new image directly from the call
    if input_path is not None:
        input_path = input_path.split('/')
        IMG_NAME = input_path[-1]
        IMG_DIR = "/".join(input_path[:-1])+"/"

    file = os.path.join(IMG_DIR, f"{IMG_NAME}")
    if not os.path.exists(file):
        print("File not found.  Exiting...")
        return

    # Load an color image
    source = cv2.imread(file, cv2.IMREAD_COLOR)
    print("Source image " + str(source.shape))

    cv2.imshow('source', source)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Swapping the channels
    output = source
    rows, cols, nbchannels = source.shape

    for i in range(rows):
        for j in range(cols):
            color = source[i, j]
            color2 = [color[2], color[1], color[0]]
            output[i, j] = color2

    cv2.imshow('output', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Allow the choice for the save path directly from the call
    print("Saving the output")
    OUT_DIR = "./output/"
    OUT_NAME = "im1.png"
    if output_path is not None:
        output_path = ("./"+output_path).split('/')
        OUT_NAME = output_path[-1]
        OUT_DIR = "/".join(output_path[:-1])+"/"
        print(OUT_DIR, OUT_NAME)
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
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
    args = parser.parse_args()

    main_seam(args.input, args.output)
