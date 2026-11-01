# Image Similarity Score

To compare an image taken at the end of a repositioning run with the stored target image, a
two-stage score is computed with `repositioning_score.py`. It combines how similar the two images
*look* (global appearance) with how well their content *lines up* (local geometry). Every value is
in [0, 1], where 1 means identical.

## Stage 1: global appearance (`global`)

Each image is passed through DINOv2 (ViT-B/14, 448 px input), a self-supervised vision transformer.
Its class-token embedding summarises the content of the whole image. The score is the cosine
similarity of the two embeddings, clipped to [0, 1]:

$$ s_{\text{global}} = \max\!\left(0,\ \frac{\mathbf{g}_T \cdot \mathbf{g}_Q}{\lVert \mathbf{g}_T \rVert \, \lVert \mathbf{g}_Q \rVert}\right) $$

It answers "is this the same scene?". It is robust to small viewpoint changes, so on its own it
can't tell a well-aligned view from a slightly offset one.

## Stage 2: local geometry

Up to 2048 SuperPoint keypoints are detected in each image at its full resolution (no resizing,
so keypoint positions and all pixel distances below are in the image's own pixels) and matched
with LightGlue. Wrong matches are removed by estimating a fundamental matrix with MAGSAC RANSAC (1.5 px
threshold, 0.999 confidence). The matches that remain are the **inliers** (`inliers`):
correspondences consistent with a single rigid viewpoint change. If fewer than 15 inliers remain,
the images are treated as not matching, and the geometric scores are set to 0.

**Overlap (`overlap`).** How much of the target scene is observed again:

$$ s_{\text{overlap}} = \min\!\left(1,\ \frac{N_{\text{inliers}}}{\min(N_T, N_Q)}\right) $$

where $N_T$ and $N_Q$ are the numbers of keypoints detected in each image.

**Normalised displacement (`median_disp_norm`).** For each inlier, its position in each image is
divided by that image's width and height, giving coordinates in [0, 1]. The displacement is the
distance between the two normalised positions. The median over all inliers is taken:

$$ d = \operatorname{median}_i \left\lVert \frac{\mathbf{p}_{T,i}}{(W_T, H_T)} - \frac{\mathbf{p}_{Q,i}}{(W_Q, H_Q)} \right\rVert $$

$d = 0$ means every matched point is at the same place in both images. $d = 0.05$ means the points
are shifted by about 5 % of the frame.

**Alignment (`alignment`).** The displacement mapped linearly to a score, reaching 0 at a
displacement of 25 % of the frame ($d_{\max} = 0.25$):

$$ s_{\text{align}} = \max\!\left(0,\ 1 - \frac{d}{d_{\max}}\right) $$

This is the component most sensitive to a remaining rotation or position error.

## Combined score (`combined`): the overall similarity

A weighted mean of the components:

$$ s = \frac{w_g\, s_{\text{global}} + w_o\, s_{\text{overlap}} + w_a\, s_{\text{align}}}{w_g + w_o + w_a}, \qquad (w_g, w_o, w_a) = (0.3,\ 0,\ 0.7) $$

The overlap term has weight 0. The fraction of keypoints that can be re-matched depends heavily on
lighting and weather, not only on the pose. Overlap is still computed and reported as a diagnostic.
In the results, *image score* refers to this combined score. Its change from the start image to the
end image is the gain.

## Pixel error

The scores above are unitless. To have a measure that is physically interpretable and comparable
across all data sets, the **pixel error** (`px`) is also reported. It is the median distance, in
pixels, between the inlier matches of the target and the compared image:

$$ e_{\text{px}} = \operatorname{median}_i \left\lVert \mathbf{p}_{T,i} - \mathbf{p}_{Q,i} \right\rVert $$

It is the same quantity as the normalised displacement $d$, expressed in pixels instead of as a
fraction of the frame. All images, simulated and real, are 1920 × 1080, so the value has the same
meaning in every data set. It is also in the same unit as the controller's stopping threshold
(1.5 px). It is undefined (not reported) when fewer than 15 inliers remain.

## Preprocessing for the real-drone images

- **Overlay masking.** The drone's video stream has a telemetry overlay (a top bar, a top-right box
  and a bottom-right box). It appears in the start and end images but not in the target image.
  Matches with a keypoint inside these regions, in either image, are discarded before RANSAC so
  that the overlay can't create false correspondences.
- **Truncated images.** Some images were only partially copied from the drone, and their lower part
  is missing. The compared images are cropped to the rows that are valid in all of them.
