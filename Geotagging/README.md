# Background

The primary objective of the Geotagging module is to accurately determine and report the real-world GPS coordinates (Latitude and Longitude) of a target detected by the drone's camera. 

# Analysis

I was going through article on this: https://drive.google.com/file/d/1iws9ud6fTy9iFwH79iYapz9Yx1tOycSB/view?usp=sharing

In the section 4.1 they have explained about the algo: 

We will need atleast:

-Drone’s gps

-Drone’s altitude above ground

-Drone Yaw,Pitch,Roll

-Target's coordinates (Like consider for now the middle of bounding box)

-Cameras Field of view

-resolution

# Recommendations

I think we can follow the approach they used in the article and modify accordingly to our needs:
