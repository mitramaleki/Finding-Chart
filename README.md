

# Astronomy Finding Chart Tool

## Overview

This tool helps you create finding charts for astronomical observations. It fetches images, processes them, and annotates with star positions and relevant information.

## Features

- Fetches finding chart images based on RA and Dec using Aladin HiPS2FITS.
- Enhances image contrast.
- Queries Gaia catalog for star data.
- Converts RA and Dec to pixel coordinates for annotation.
- Displays images with annotations and interactive tooltips.

## Installation

Install the required packages:

```sh
pip install requests Pillow matplotlib mplcursors numpy astroquery astropy
```

## Usage

1. Clone the repository:

```sh
git clone https://github.com/yourusername/astronomy-finding-chart-tool.git
cd astronomy-finding-chart-tool
```

2. Run the script:

```sh
python main.py
```

3. Enter the required information when prompted:

- Observing run ID
- PI Name
- OB Name
- Target Name
- RA (in degrees)
- Dec (in degrees)
- Wavelength Range
- Scale Length (in arcseconds)
- Field of View (FOV) in arcminutes

You can also add RA and Dec offsets if needed.

## Example

```
Enter the Observing run ID: 12345
Enter the PI Name: John Doe
Enter the OB Name: Observation 1
Enter the Target Name: Alpha Centauri
Enter the RA (in degrees): 219.9021
Enter the Dec (in degrees): -60.8339
Enter the Wavelength Range: 400-700 nm
Enter the Scale Length (in arcseconds): 120
Enter the Field of View (FOV) in arcminutes (default 20): 4
Enter the RA Offset (in arcseconds, or 'done' to finish): 218.80
Enter the Dec Offset (in arcseconds): -60.80
Enter the RA Offset (in arcseconds, or 'done' to finish): done
```

## How It Works

- `fetch_aladin_finding_chart`: Fetches an image from Aladin HiPS2FITS.
- `enhance_contrast`: Enhances image contrast.
- `fetch_star_data`: Queries Gaia catalog for star data.
- `ra_dec_to_pixels`: Converts RA and Dec to pixel coordinates.
- `display_image_with_annotations`: Displays the image with annotations.

## License

This project is for INO(Iran National Observatory).
