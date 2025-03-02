# Liquid droplets and salt crystals detection and tracking


Input: Video file

Condensation process:
- Each frame demonstrates the same tube channel (snake-looking) with the gas flowing through.
- Due to changes in pressure and humidity, there are several condensation effects: 
    * the gas condensation starts to appear on the tube walls
    * the salt matter condenses on the walls, effectively closing the tube channel

## Goal: to detect the droplets of liquid that appear. Detect the salt regions and quantify the amount of salt that has condensed.

____________________________________________

### Pre-processing 
1) Cut the video frame to localize the tube
2) Align the frames (if possible)
3) Prepare gray version (no useful info in colors)
4) Prepare the template of the tube channel
5) Mask out all non-tube regions

All can be reproduced in `preproc.py`.

### Method

1) Compute diffs: changes from frame i to frame i+1
2) Threshold: remove background noise (due to color jittering when shot); thresholding values may vary

Example of thresholding on different levels can be seen in `data/processed/thresholded/T_stack.mp4`.

> NOTE: Video with T multiplier = 3 `Tmult_3.mp4` is the most illustrative, continue with it.

### TODO

3) Detect a change. Init the clusters of growth. Follow the changes in neighboring pixels (filtering out minor flukes). 


NOTE: This project on [Crystal Growth Rate Analysis](https://github.com/jsbangsund/crystal-growth-rate-analysis) seems related. Are we interested?
