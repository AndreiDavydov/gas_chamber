# Liquid droplets and salt crystals detection and tracking


Input: Video file

Condensation process:
- Each frame demonstrates the same tube channel (snake-looking) with the gas flowing through.
- Due to changes in pressure and humidity, there are several condensation effects: 
    * the gas condensation starts to appear on the tube walls
    * the salt matter condenses on the walls, effectively closing the tube channel

## Goal: to detect the droplets of liquid that appear. Detect the salt regions and quantify the amount of salt that has condensed.

____________________________________________

### Plan -- pre-processing 
1) Cut the video frame to localize the tube
2) Align the frames (if possible)
3) Prepare gray version (no useful info in colors)
4) Prepare the template of the tube channel
5) Mask out all non-tube regions

### TODO