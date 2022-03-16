# SeamCarving

The seam carving project proposed our implementation of the Seam Carving and of some of its coolest extension.
We will also try to find new exiting tools based on SeamCarving.

### For videos

- seam-video-original.pdf
A seam is a monotonic and connected path of pixels running from top to bottom.
If each image was treated independently, it would move within the image and content would appear to move around within the frame.
Seam must be contiguous between frames to prevent jitter.
First approach (static) - consider gradients in x,y,t dimension, then $E(i,j)= \alpha*E_spatial + (1-\alpha)*E_temporal$
Appeal - simplicity and speed.

Dynamic programming -> Graph cuts
Using infinite weight arcs, we can impose monotonicity and connectedness constraints

Cannot solve min-cut exactly, but can use a banded multiresoluational model.  Approximate cut on a coarse-grained graph, then done again for higher resolutions.

(Artifacts in images because we cut out the lowest energy seam, but we add in energy from the join between the two sides, and dont include this in the calculation)

Members of the team:
- Maxime Cautrès
- Thomas Pickles

Supervisor:
- David Coeurjolly
- Vincent Nivoliers

TODO:
[] Have a look at Pillow [PIL.ImageFilter](https://pillow.readthedocs.io/en/stable/reference/ImageFilter.html) library.  A whole range of different filters and image processing tools available there, and they _may be_ (I haven't checked this) optimised for faster execution.

Done:
[x] Implementing the seam carving for picture
[x] Implementing a better autosave - option to overwrite (-f), or will write to a new timestamped path.

### How to run code

From `src/` folder, type `python3 main.py -h` for explanation of command-line options