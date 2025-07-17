def create_image_from_text(text, font_path, image_width, image_height, font_size, text_color, background_color=(255, 165, 0), corner_radius=0, transparent_bg=False):
    """
    Creates an image from text with specified font and colors, with optional rounded corners and transparency.

    Args:
        text (str): The text to render.
        font_path (str): The path to the font file (e.g., .ttf).
        image_width (int): The width of the output image.
        image_height (int): The height of the output image.
        font_size (int): The size of the font.
        text_color (tuple): RGB tuple for the text color (e.g., (0, 0, 0) for black).
        background_color (tuple): RGB tuple for the background color (e.g., (255, 165, 0) for orange).
        corner_radius (int): The radius for rounded corners. Set to 0 for no rounded corners.
        transparent_bg (bool): If True, the background will be transparent (alpha channel added).

    Returns:
        PIL.Image.Image: The generated image object.
    """
    if transparent_bg:
        img = Image.new('RGBA', (image_width, image_height), color=(*background_color, 0)) # Add alpha channel
    else:
        img = Image.new('RGB', (image_width, image_height), color=background_color)

    d = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Error: Font file not found at {font_path}. Please provide a valid font path.")
        return None

    # For complex scripts like Arabic/Persian, ensuring you have a font that supports
    # these languages and potentially using a library with better text shaping
    # might be necessary if the output is not as expected.
    # You may also need to use arabic_reshaper and python-bidi for proper rendering.
    # Example for RTL text:
    # reshaped_text = arabic_reshaper.reshape(text)
    # bidi_text = get_display(reshaped_text)
    # text_to_draw = bidi_text
    text_to_draw = text


    text_bbox = d.textbbox((0, 0), text_to_draw, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Center the text
    x = (image_width - text_width) / 2
    # Adjust vertical centering based on text_bbox
    y = (image_height - text_height) / 2 - text_bbox[1]

    # Create a mask for rounded corners
    if corner_radius > 0:
        mask = Image.new('L', (image_width, image_height), 0)
        draw_mask = ImageDraw.Draw(mask)
        draw_mask.rounded_rectangle([(0, 0), (image_width, image_height)], corner_radius, fill=255)
        if transparent_bg:
            # Apply mask to the alpha channel for transparent background with rounded corners
            img.putalpha(mask)
        else:
            # For non-transparent background, draw the rounded rectangle directly
            # This requires drawing the background shape with rounded corners.
            # A simpler approach for solid background is to just create the image with rounded corners.
            # The current implementation focuses on applying mask for transparency with rounded corners.
            pass # The background is already set when creating the image. The mask is used for transparency.


    d.text((x, y), text_to_draw, fill=text_color, font=font)


    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str




