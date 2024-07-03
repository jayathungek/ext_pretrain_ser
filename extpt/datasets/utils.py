import colorsys
from typing import List


def get_label_colours(n: int) -> List[int]:
    colors = []
    hue_step = 360.0 / n

    for i in range(n):
        hue = i * hue_step
        saturation = 1  # You can adjust saturation and lightness if needed
        lightness = 0.4   # You can adjust saturation and lightness if needed

        rgb = colorsys.hls_to_rgb(hue / 360.0, lightness, saturation)
        hexcol = "#" + "".join([f"{int(v * 255):02X}" for v in rgb])
        colors.append(hexcol)

    return colors
