import numpy as np
import cv2
import os

IMG_DIR = "./images/"
IMG_NAME = "sunset"


def main_seam():
    file = os.path.join(IMG_DIR, f"{IMG_NAME}.png")
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

    print("Saving the output")
    cv2.imwrite('output.png', output)


if __name__ == '__main__':
    main_seam()
