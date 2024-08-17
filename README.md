# 🌟 Star Finder and Astrometric Plotter

Hi! Welcome to my project. This script helps you fetch celestial images, detect stars, match them with Gaia catalog data, and display interactive plots. It’s designed for astronomers, researchers, or anyone who's into exploring the stars. You can input your target's RA/Dec, and it will handle the rest—fetching images, detecting stars, matching them with the Gaia catalog, and drawing everything on a nice annotated plot. 

## ✨ What It Can Do

- **Get Images from the Sky**: I’m using Aladin's HiPS2FITS service to pull down images based on RA/Dec coordinates.
- **Detect Stars**: The script can detect stars in the image using OpenCV and some thresholding magic.
- **Match Stars with Gaia Data**: It pulls star data from the Gaia catalog and matches the detected stars to actual stars with their RA, Dec, and magnitudes.
- **Display Interactive Plots**: You’ll get a plot showing all the stars with a field of view circle, star markers, and directional labels (N, S, E, W). You can even hover over the stars for more details like magnitude and position.
- **Customizable Settings**: You can adjust the field of view, scale bars, position angles, and more to fit your specific needs.

## 🛠️ Setup

You’ll need Python 3.7 or above. Here’s how to install everything you need:

pip install requests Pillow matplotlib numpy opencv-python astroquery astropy mplcursors


Example:
if __name__ == "__main__":
    main(
        observing_run_id='12345',
        pi_name='Dr. Jane Doe',
        ob_name='Observation XYZ',
        target='Target Name',
        ra=103.34,
        dec=23.45,
        wavelength_range=323,
        scale_length_input=50,
        fov_arcminutes_input=4,
        ra_offsets=[50],
        dec_offsets=[50],
        pos_angle=45
    )
