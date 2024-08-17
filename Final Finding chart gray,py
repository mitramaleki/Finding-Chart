#This Code is Written by Mitra Maleki
import requests
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import mplcursors
import io
import numpy as np
import cv2
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

# Fetch the finding chart image based on Right Ascension (RA) and Declination (Dec)
# using the Aladin HiPS2FITS service. Returns the image and the pixel scale.
def fetch_aladin_finding_chart(ra, dec, width=512, height=512, fetch_fov_degrees=1.0):
    base_url = "https://alasky.u-strasbg.fr/hips-image-services/hips2fits"
    params = {
        'hips': 'P/DSS2/color',  # Request DSS2 color image
        'ra': ra,                # RA in degrees
        'dec': dec,              # Dec in degrees
        'fov': fetch_fov_degrees, # Field of view in degrees
        'width': width,          # Image width in pixels
        'height': height,        # Image height in pixels
        'projection': 'TAN',     # Use TAN projection
        'format': 'jpg'          # Request a JPEG image
    }
    response = requests.get(base_url, params=params)
    if response.status_code != 200:
        response.raise_for_status()
    # Open the image as grayscale and convert it to a PIL Image
    image = Image.open(io.BytesIO(response.content)).convert("L")
    pixel_scale = compute_pixel_scale(fetch_fov_degrees, width)  # Calculate pixel scale
    return image, pixel_scale

# Compute the pixel scale in arcseconds per pixel based on the field of view and image width.
def compute_pixel_scale(fov_degrees, width):
    return (fov_degrees * 3600) / width  # Convert FOV from degrees to arcseconds and divide by image width

# Find stars in the image based on their brightness using contour detection.
# Returns a list of stars (x, y, radius) that meet the threshold criteria.
def find_stars(image, threshold=200, cluster_threshold=100):
    image_array = np.array(image)  # Convert image to numpy array
    _, binary_image = cv2.threshold(image_array, threshold, 255, cv2.THRESH_BINARY)  # Apply threshold to image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours
    
    stars = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)  # Get the enclosing circle for each contour
        area = cv2.contourArea(contour)  # Calculate the contour area
        
        # Filter out small objects and large clusters
        if radius > 1 and area <= cluster_threshold * 10:
            stars.append((int(x), int(y), int(radius)))  # Add the star's position and size to the list
    
    return stars

# Fetch star data from the Gaia catalog in a bulk query using a cone search around the target RA/Dec.
def fetch_star_data_bulk(target_ra, target_dec, fetch_radius_degrees):
    coord = SkyCoord(ra=target_ra, dec=target_dec, unit=(u.deg, u.deg), frame='icrs')  # Set up the target coordinates
    radius = fetch_radius_degrees * u.deg  # Convert the radius to astropy units
    j = Gaia.cone_search_async(coord, radius=radius)  # Perform asynchronous cone search
    r = j.get_results()  # Get the query results
    
    star_data = []
    for star in r:
        # Collect necessary data for each star
        star_data.append({
            'ra': star['ra'],  # Right Ascension
            'dec': star['dec'],  # Declination
            'mag': star['phot_g_mean_mag']  # Magnitude in G band
        })

    return star_data

# Convert pixel coordinates to RA and Dec using the pixel scale and the image center.
def pixel_to_ra_dec(x, y, target_ra, target_dec, pixel_scale, image_width, image_height):
    delta_ra = (x - image_width / 2) * pixel_scale / 3600  # Calculate the RA offset from the center
    delta_dec = -(y - image_height / 2) * pixel_scale / 3600  # Calculate the Dec offset from the center

    # Adjust the RA based on the target Dec
    ra = target_ra + delta_ra / np.cos(np.deg2rad(target_dec))
    dec = target_dec + delta_dec

    return ra, dec

# Convert RA and Dec offsets to pixel offsets for drawing on the image.
def ra_dec_offsets_to_pixels(ra_offset, dec_offset, pixel_scale, image_width, image_height):
    x_offset_pixels = ra_offset / pixel_scale  # Convert RA offset to pixels
    y_offset_pixels = -dec_offset / pixel_scale  # Convert Dec offset to pixels (invert due to image coordinate system)
    return x_offset_pixels, y_offset_pixels

# Match detected stars in the image with Gaia catalog data based on position.
def match_stars_to_gaia(stars, gaia_data, target_ra, target_dec, pixel_scale, image_width, image_height):
    matched_stars = []
    for (x, y, radius) in stars:
        # Convert star's pixel coordinates to RA/Dec
        ra, dec = pixel_to_ra_dec(x, y, target_ra, target_dec, pixel_scale, image_width, image_height)
        
        # Find the closest Gaia star based on position
        closest_star = min(gaia_data, key=lambda star: np.hypot(ra - star['ra'], dec - star['dec']))
        
        # Add matched star data to the list
        matched_stars.append({
            'x': x,
            'y': y,
            'radius': max(radius, 3),  # Ensure a minimum circle size for visibility
            'ra': closest_star['ra'],
            'dec': closest_star['dec'],
            'mag': closest_star['mag']  # Magnitude from Gaia data
        })
    
    return matched_stars

# Draw a blue plus marker at the center of the field of view.
def draw_center_target(ax, x, y, color='blue'):
    size = 30  # Length of the crosshair
    ax.plot([x - size, x + size], [y, y], color=color, lw=1)  # Horizontal line of the crosshair
    ax.plot([x, x], [y - size, y + size], color=color, lw=1)  # Vertical line of the crosshair

# Draw a rotating red square inside the field of view (FOV) circle.
def draw_rotating_square(ax, center_x, center_y, radius, angle, color='red'):
    square_size = radius / np.sqrt(2)  # Size of the square (half the diagonal of the FOV circle)
    
    # Define square corners and apply rotation
    corners = np.array([
        [-square_size, -square_size],
        [square_size, -square_size],
        [square_size, square_size],
        [-square_size, square_size]
    ])
    rotation_matrix = np.array([
        [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle))],
        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle))]
    ])
    rotated_corners = np.dot(corners, rotation_matrix) + [center_x, center_y]
    
    # Create a polygon and add it to the plot
    square = patches.Polygon(rotated_corners, closed=True, edgecolor=color, facecolor='none', linewidth=1)
    ax.add_patch(square)

# Draw a circular marker for detected stars on the image.
def draw_star_marker(ax, x, y, radius, color):
    circle = patches.Circle((x, y), radius, edgecolor=color, facecolor='none', lw=1)  # Create a circle patch
    ax.add_patch(circle)  # Add circle to the plot

# Add a scale bar to the image to show distance in arcseconds.
def add_scale_bar(ax, pixel_scale, image_height):
    scale_length = 50  # Length of the scale bar in arcseconds
    scale_bar_length_pixels = scale_length / pixel_scale  # Convert arcseconds to pixels

    # Position the scale bar near the bottom-left corner of the image
    scale_bar_start = (10, image_height - 50)
    scale_bar_end = (10 + scale_bar_length_pixels, image_height - 50)
    
    # Draw the scale bar as a white line
    ax.plot([scale_bar_start[0], scale_bar_end[0]], [scale_bar_start[1], scale_bar_end[1]], color='white', lw=2)
    
    # Add a label above the scale bar showing its length in arcseconds
    ax.text((scale_bar_start[0] + scale_bar_end[0]) / 2, scale_bar_start[1] - 10, f"{int(scale_length)}\"", color="white", ha='center')

# Display the image with detected stars, data, and annotations (e.g., FOV, offsets, etc.)
def display_image_with_detected_stars_and_data(image, matched_stars, target_ra, target_dec, pixel_scale, info_text, offsets, pos_angle, display_fov_radius_pixels):
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image, cmap='gray')
    ax.axis('off')  # Hide axes

    width, height = image.size

    # Draw FOV circle in red
    fov_circle = patches.Circle((width / 2, height / 2), display_fov_radius_pixels, linewidth=1, edgecolor='red', facecolor='none')
    ax.add_patch(fov_circle)

    # Draw the rotating red square inside the FOV circle
    draw_rotating_square(ax, width / 2, height / 2, display_fov_radius_pixels, pos_angle)

    # Draw star markers for detected stars
    for star in matched_stars:
        x, y, radius = star['x'], star['y'], star['radius']
        draw_star_marker(ax, x, y, radius, 'red')

    # Draw the center target marker
    draw_center_target(ax, width // 2, height // 2, "blue")

    # Plot the RA/Dec offsets (if any)
    offset_scatter = []
    if offsets:
        for offset in offsets:
            ra_offset_pixels, dec_offset_pixels = ra_dec_offsets_to_pixels(offset['ra_offset'], offset['dec_offset'], pixel_scale, width, height)
            offset_x = width // 2 + ra_offset_pixels
            offset_y = height // 2 + dec_offset_pixels
            offset_scatter.append((offset_x, offset_y, target_ra + offset['ra_offset'] / 3600, target_dec + offset['dec_offset'] / 3600))

    # Draw scatter points for offsets
    if offset_scatter:
        offset_x_coords, offset_y_coords, _, _ = zip(*offset_scatter)
        ax.scatter(offset_x_coords, offset_y_coords, color='yellow', s=50, marker='+')

    # Add cardinal direction labels (N, S, E, W)
    ax.text(width / 2, 20, "N", color="white", ha='center', va='top', fontsize=15)
    ax.text(width / 2, height - 20, "S", color="white", ha='center', va='bottom', fontsize=15)
    ax.text(20, height / 2, "E", color="white", va='center', ha='left', fontsize=15)
    ax.text(width - 20, height / 2, "W", color="white", va='center', ha='right', fontsize=15)

    # Add interactive tooltips for detected stars and target/offset points
    all_scatter = matched_stars + [(width // 2, height // 2, target_ra, target_dec, 'Target')] + [(offset[0], offset[1], offset[2], offset[3], 'Offset') for offset in offset_scatter]
    
    if all_scatter:
        scatter_x, scatter_y, scatter_ra, scatter_dec, scatter_info = zip(*[(s['x'], s['y'], s['ra'], s['dec'], s.get('mag', 'Target')) for s in matched_stars] + [(width // 2, height // 2, target_ra, target_dec, 'Target')] + [(o[0], o[1], o[2], o[3], 'Offset') for o in offset_scatter])

        # Convert RA and Dec to hh:mm:ss and dd:mm:ss formats
        scatter_ra_hms = [degrees_to_ra(ra) for ra in scatter_ra]
        scatter_dec_dms = [degrees_to_dec(dec) for dec in scatter_dec]

        scatter_plot = ax.scatter(scatter_x, scatter_y, color='none', s=10, alpha=0)  # Create invisible scatter plot for tooltips
        cursor = mplcursors.cursor(scatter_plot, hover=True)  # Enable interactive tooltips
        cursor.connect("add", lambda sel: sel.annotation.set_text(f"RA: {scatter_ra_hms[sel.index]}\nDec: {scatter_dec_dms[sel.index]}\n{'Mag: ' + str(scatter_info[sel.index]) if scatter_info[sel.index] != 'Target' and scatter_info[sel.index] != 'Offset' else scatter_info[sel.index]}"))

    # Add the scale bar to the bottom-left of the image
    add_scale_bar(ax, pixel_scale, height)

    # Add title text with observing run information
    plt.title(info_text)

    # Show the image with annotations
    plt.show()

# Convert degrees to Right Ascension in hh:mm:ss format
def degrees_to_ra(deg):
    hours = int(deg / 15)  # Convert degrees to hours
    minutes = int((deg / 15 - hours) * 60)  # Convert fractional hours to minutes
    seconds = ((deg / 15 - hours) * 60 - minutes) * 60  # Convert fractional minutes to seconds
    return f"{hours:02d}:{minutes:02d}:{seconds:.2f}"

# Convert degrees to Declination in dd:mm:ss format
def degrees_to_dec(deg):
    sign = '-' if deg < 0 else '+'  # Handle negative declinations
    deg = abs(deg)
    degrees = int(deg)
    arcminutes = int((deg - degrees) * 60)
    arcseconds = ((deg - degrees) * 60 - arcminutes) * 60
    return f"{sign}{degrees:02d}:{arcminutes:02d}:{arcseconds:.2f}"

# Main function to run the full application with user-supplied parameters
def main(observing_run_id, pi_name, ob_name, target, ra, dec, wavelength_range, scale_length_input, fov_arcminutes_input, ra_offsets, dec_offsets, pos_angle):
    scale_length = float(scale_length_input) if scale_length_input else 50  # Default scale length is 50 if not provided
    fov_arcminutes = float(fov_arcminutes_input) if fov_arcminutes_input else 20  # Default FOV is 20 arcminutes

    # Convert RA and Dec from degrees to hh:mm:ss and dd:mm:ss formats
    ra_hms = degrees_to_ra(ra)
    dec_dms = degrees_to_dec(dec)

    # Collect RA and Dec offsets
    offsets = []
    if ra_offsets and dec_offsets and len(ra_offsets) == len(dec_offsets):
        for ra_offset, dec_offset in zip(ra_offsets, dec_offsets):
            offsets.append({'ra_offset': ra_offset, 'dec_offset': dec_offset})

    # Define the fetching FOV (always larger than the display FOV)
    fetch_fov_arcminutes = max(10, fov_arcminutes + 6)
    fetch_fov_degrees = fetch_fov_arcminutes / 60

    # Fetch the image with the larger FOV
    image, pixel_scale = fetch_aladin_finding_chart(ra, dec, 512, 512, fetch_fov_degrees)

    # Calculate the display FOV radius in pixels
    display_fov_radius_pixels = (fov_arcminutes * 60) / pixel_scale / 2

    # Detect stars in the image
    stars = find_stars(image)

    # Fetch Gaia star data for the larger FOV
    gaia_data = fetch_star_data_bulk(ra, dec, fetch_fov_degrees)

    # Match detected stars with Gaia data
    matched_stars = match_stars_to_gaia(stars, gaia_data, ra, dec, pixel_scale, image.width, image.height)

    # Prepare info text for the plot
    info_text = f"run ID: {observing_run_id} | PI: {pi_name} | OB: {ob_name} | Target: {target}\nRA: {ra_hms} | Dec: {dec_dms} | Wavelength: {wavelength_range} | Scale: {scale_length}"

    # Display the image with annotations
    display_image_with_detected_stars_and_data(image, matched_stars, ra, dec, pixel_scale, info_text, offsets, pos_angle, display_fov_radius_pixels)

# If this script is run directly, feel free to put your own inputs.(=
if __name__ == "__main__":
    main('observing_run_id', 'pi_name', 'ob_name', 'target', 103.34, 23.45, 323, 50, 4, [50], [50], 45)
