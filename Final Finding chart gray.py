import requests  # for making HTTP requests to fetch images and star data
from PIL import Image, ImageEnhance  # for image processing
import matplotlib.pyplot as plt  # for displaying images with annotations
import matplotlib.patches as patches  # for drawing shapes on Matplotlib axes
import mplcursors  # for interactive tooltips
import io  # for handling in-memory byte streams
import numpy as np  # for numerical operations
from astroquery.gaia import Gaia  # for querying Gaia catalog which find our stars magnitude
from astropy.coordinates import SkyCoord  # for coordinate transformations
import astropy.units as u  # for unit conversions

# to fetch the finding chart image based on RA and Dec using Aladin HiPS2FITS
def fetch_aladin_finding_chart(ra, dec, width=512, height=512, fov_degrees=1.0):
    base_url = "https://alasky.u-strasbg.fr/hips-image-services/hips2fits"
    params = {
        'hips': 'P/DSS2/color',  # Survey
        'ra': ra,  # Right Ascension
        'dec': dec,  # Declination
        'fov': fov_degrees,  # Field of View in degrees
        'width': width,  # Image width in pixels
        'height': height,  # Image height in pixels
        'projection': 'TAN',  # Projection type
        'format': 'jpg'  # Image format
    }
    response = requests.get(base_url, params=params)  # Send HTTP get request to fetch the image
    if response.status_code != 200:
        response.raise_for_status()  # Raise an error if the request was unsuccessful
    image = Image.open(io.BytesIO(response.content)).convert("L")  # Open the image from the response content and convert to grayscale
    pixel_scale = compute_pixel_scale(fov_degrees, width)  # compute the pixel scale (arcseconds per pixel)
    return image, pixel_scale

# to compute the pixel scale based on the field of view and image width
def compute_pixel_scale(fov_degrees, width):
    return (fov_degrees * 3600) / width  # convert field of view from degrees to arcseconds and divide by image width

# to enhance the contrast of an image
def enhance_contrast(image):
    enhancer = ImageEnhance.Contrast(image)  # contrast enhancer
    return enhancer.enhance(2)  # Increase contrast by a factor of 2 more is not needed

# to fetch star data from Gaia catalog cause they have a compelete data base
def fetch_star_data(ra, dec, fov_degrees):
    coord = SkyCoord(ra=ra, dec=dec, unit=(u.deg, u.deg), frame='icrs')  # create a SkyCoord object for the target position
    radius = fov_degrees / 2 * u.deg  # compute our telescope radius in degrees
    j = Gaia.cone_search_async(coord, radius=radius)  # a cone search in the Gaia catalog
    r = j.get_results()  # search results
    
    # first few rows of the result for debugging
    print("Star data from Gaia query:")
    print(r[:10])
    
    # Check if 'phot_g_mean_mag' is available and print data
    if 'phot_g_mean_mag' in r.colnames:
        print("Magnitude data for the first 10 stars:")
        print(r['phot_g_mean_mag'][:10])
    else:
        print("Magnitude data ('phot_g_mean_mag') not found in the query results.")
    
    # Filter stars with magnitude brighter than 16
    star_data = [{'ra': star['ra'], 'dec': star['dec'], 'mag': star['phot_g_mean_mag']} for star in r if star['phot_g_mean_mag'] < 16]
    
    # filtered star data for debugging and further use
    print("Filtered star data:")
    print(star_data)
    
    return star_data

# to convert RA, Dec to pixel coordinates
def ra_dec_to_pixels(ra, dec, image_center_ra, image_center_dec, pixel_scale, image_width, image_height):
    delta_ra = (ra - image_center_ra) * 3600 / pixel_scale  # compute the RA offset in pixels
    delta_dec = (dec - image_center_dec) * 3600 / pixel_scale  # compute the Dec offset in pixels
    x = image_width / 2 + delta_ra  # compute the x-coordinate in the image
    y = image_height / 2 - delta_dec  # compute the y-coordinate in the image
    return x, y

# to draw a target marker in colors
def draw_target(ax, x, y, color):
    size = 10  # Size of the target marker
    ax.plot([x - size, x + size], [y, y], color=color, lw=1)  # draw horizontal line
    ax.plot([x, x], [y - size, y + size], color=color, lw=1)  # draw vertical line

# to check if a point is within a circle
def is_within_fov(x, y, circle_x, circle_y, radius):
    return (x - circle_x) ** 2 + (y - circle_y) ** 2 <= radius ** 2  # Check if the point (x, y) is within the circle

# to display the image with annotations
def display_image_with_annotations(image, annotations, fov_radius_pixels, info_text, scale_length, pixel_scale, target, offsets):
    image = enhance_contrast(image)  # Enhance image contrast
    image = image.rotate(180)  # Rotate the image for correct telescope view

    width, height = image.size  # get image dimensions

    fig, ax = plt.subplots()  # create a Matplotlib figure and axis
    ax.imshow(image, cmap='gray')  # display the image in grayscale

    # FOV circle
    fov_circle = patches.Circle((width / 2, height / 2), fov_radius_pixels, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(fov_circle)  # add the FOV circle to the axis

    # cardinal directions, white is more efficient as our image is grayscale
    ax.text(width / 2, 20, "N", color="white", ha='center', va='top', fontsize=15)  # North label
    ax.text(width / 2, height - 20, "S", color="white", ha='center', va='bottom', fontsize=15)  # South label
    ax.text(20, height / 2, "E", color="white", va='center', ha='left', fontsize=15)  # East label
    ax.text(width - 20, height / 2, "W", color="white", va='center', ha='right', fontsize=15)  # West label

    # Calculate tick interval and scale bar length in pixels
    tick_interval = scale_bar_length_pixels = scale_length / pixel_scale
    if scale_bar_length_pixels > width - 20:  # Adjust if the scale bar length exceeds image width
        scale_bar_length_pixels = width - 20
        scale_length = scale_bar_length_pixels * pixel_scale

    x_ticks = np.arange(0, width, tick_interval)  # set x-axis tick positions
    y_ticks = np.arange(0, height, tick_interval)  # set y-axis tick positions
    x_labels = (x_ticks * pixel_scale).astype(int)  # convert x-tick positions to arcseconds
    y_labels = (y_ticks * pixel_scale).astype(int)  # convert y-tick positions to arcseconds

    ax.set_xticks(x_ticks)  # apply x-axis ticks to the axis
    ax.set_xticklabels([])  # remove x-axis tick labels
    ax.set_yticks(y_ticks)  # apply y-axis ticks to the axis
    ax.set_yticklabels([])  # remove y-axis tick labels

    scatter = []
    for annotation in annotations:  # working through each annotation
        x, y = ra_dec_to_pixels(annotation['ra'], annotation['dec'], annotation['center_ra'], annotation['center_dec'], pixel_scale, width, height)
        if is_within_fov(x, y, width / 2, height / 2, fov_radius_pixels):  # Check if they are within the FOV
            draw_target(ax, x, y, "red")  # draw red target marker for each star
            scatter.append((x, y, annotation['ra'], annotation['dec'], annotation['mag']))  # adding RA, Dec, and magnitude

    # draw the target marker in blue
    target_x, target_y = ra_dec_to_pixels(target['ra'], target['dec'], target['center_ra'], target['center_dec'], pixel_scale, width, height)
    draw_target(ax, target_x, target_y, "blue")  # draw blue target marker for the main target to be different from stars

    # draw offset markers
    offset_colors = ['green', 'yellow', 'cyan', 'magenta']  # colors for different offsets
    offset_scatter = []
    for i, offset in enumerate(offsets):
        offset_x, offset_y = ra_dec_to_pixels(offset['ra'], offset['dec'], target['center_ra'], target['center_dec'], pixel_scale, width, height)
        color = offset_colors[i % len(offset_colors)]  # cycle through colors if more offsets than colors
        draw_target(ax, offset_x, offset_y, color)  # draw offset marker
        offset_scatter.append((offset_x, offset_y, offset['ra'], offset['dec']))

    # Create a unified cursor for tooltips
    all_scatter = scatter + [(target_x, target_y, target['ra'], target['dec'], 'Target')]
    for offset in offset_scatter:
        all_scatter.append((offset[0], offset[1], offset[2], offset[3], 'Offset'))

    if all_scatter:
        scatter_x, scatter_y, scatter_ra, scatter_dec, scatter_info = zip(*all_scatter)  # unpack all scatter data
        scatter_plot = ax.scatter(scatter_x, scatter_y, color='none', s=10, alpha=0)  # create invisible scatter plot
        cursor = mplcursors.cursor(scatter_plot, hover=True)  # enable hover tooltips
        cursor.connect("add", lambda sel: sel.annotation.set_text(
            f"RA: {scatter_ra[sel.index]:.4f}\nDec: {scatter_dec[sel.index]:.4f}\n{'Mag' if scatter_info[sel.index] != 'Target' and scatter_info[sel.index] != 'Offset' else 'Type'}: {scatter_info[sel.index]}"))  # showing RA, Dec, and either magnitude or type on hover

    # draw scale bar without numbers
    scale_bar_start = (10, height - 50)  # Starting position of scale bar
    scale_bar_end = (10 + scale_bar_length_pixels, height - 50)  # Ending position of scale bar
    ax.plot([scale_bar_start[0], scale_bar_end[0]], [scale_bar_start[1], scale_bar_end[1]], color='white', lw=2)  # draw scale bar
    ax.text((scale_bar_start[0] + scale_bar_end[0]) / 2, scale_bar_start[1] - 10, f"{int(scale_length)}\"", color="white", ha='center')  # add scale length text

    plt.figtext(0.5, 0.95, info_text, wrap=True, horizontalalignment='center', fontsize=12)  # add information text
    plt.show()  # display the plot

# to run the application
def main():
    observing_run_id = input("Enter the Observing run ID: ")  # input for Observing run ID
    pi_name = input("Enter the PI Name: ")  # input for PI Name
    ob_name = input("Enter the OB Name: ")  # input for OB Name
    target = input("Enter the Target Name: ")  # input for Target Name
    ra = float(input("Enter the RA (in degrees): "))  # input for RA in degrees
    dec = float(input("Enter the Dec (in degrees): "))  # input for Dec in degrees
    wavelength_range = input("Enter the Wavelength Range: ")  # input for Wavelength Range
    scale_length_input = input("Enter the Scale Length (in arcseconds): ")  # input for Scale Length in arcseconds
    scale_length = float(scale_length_input) if scale_length_input else 50  # Default scale length will be 50 if not provided by user
    fov_arcminutes_input = input("Enter the Field of View (FOV) in arcminutes (default 20): ")  # input for FOV in arcminutes
    fov_arcminutes = float(fov_arcminutes_input) if fov_arcminutes_input else 20  # Default FOV will be 20 if not provided by user

    # fetch_fov_arcminutes based on the specified fov_arcminutes
    if fov_arcminutes < 6:
        fetch_fov_arcminutes = 7
    elif fov_arcminutes < 10:
        fetch_fov_arcminutes = 12
    elif fov_arcminutes < 15:
        fetch_fov_arcminutes = 17  # I believe this is enough for a good view
    else:
        fetch_fov_arcminutes = 22  # as our telescope is 20 at best we do not need to go any further

    fetch_fov_degrees = fetch_fov_arcminutes / 60  # convert FOV from arcminutes to degrees

    # apply offsets to RA and Dec
    offsets = []
    while True:
        ra_offset = input("Enter the RA Offset (in arcseconds, or 'done' to finish): ")
        if ra_offset.lower() == 'done':
            break
        ra_offset = float(ra_offset)
        dec_offset = float(input("Enter the Dec Offset (in arcseconds): "))
        ra_offset_deg = ra_offset / 3600  # convert RA offset from arcseconds to degrees
        dec_offset_deg = dec_offset / 3600  # convert Dec offset from arcseconds to degrees
        offset_ra = ra + ra_offset_deg / np.cos(np.deg2rad(dec))  # apply RA offset, adjusting for declination
        offset_dec = dec + dec_offset_deg  # apply Dec offset
        offsets.append({'ra': offset_ra, 'dec': offset_dec})

    # create information text
    info_text = f"run ID: {observing_run_id} | PI: {pi_name} | OB: {ob_name}\nTarget: {target} | RA: {ra} | Dec: {dec} | Wavelength: {wavelength_range} | Scale: {scale_length}\""

    try:
        # fetch the finding chart image using Aladin
        image, pixel_scale = fetch_aladin_finding_chart(ra, dec, fov_degrees=fetch_fov_degrees)
        actual_fov_radius_pixels = (fov_arcminutes * 60) / pixel_scale / 2  # convert FOV from arcminutes to pixels , to make it real with image pixels

        if actual_fov_radius_pixels * 2 > min(image.width, image.height):  # make sure radius is within image bounds
            actual_fov_radius_pixels = min(image.width, image.height) / 2

        # fetch star data from Gaia catalog
        star_data = fetch_star_data(ra, dec, fetch_fov_degrees)
        annotations = [{
            'ra': star['ra'],
            'dec': star['dec'],
            'mag': star['mag'],
            'center_ra': ra,
            'center_dec': dec
        } for star in star_data]

        # main target information
        target_info = {
            'ra': ra,
            'dec': dec,
            'center_ra': ra,
            'center_dec': dec
        }

        # display the image with annotations
        display_image_with_annotations(image, annotations, actual_fov_radius_pixels, info_text, scale_length, pixel_scale, target_info, offsets)
    except Exception as e:
        print(f"An error occurred: {e}")  # print any errors that occur

if __name__ == "__main__":
    main()  # run main  if the script is working correctly
