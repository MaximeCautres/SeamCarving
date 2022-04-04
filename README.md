# SeamCarving

This project allows a user to shrink an image to a given size by removing uninteresting regions from the image.

## Authors
- Maxime Cautrès
- Thomas Pickles

### How to run code

For command-line application, type `python3 algo.py -h` for explanation of command-line options.
eg. To resize an image (here a dali painting) to 90% of its original width and with a height of 300 pixels, with 5 columns removed between recomputations:
- `python3 algo.py --width=90% -height=300 --per-step=5 -i=./images/dali.png`