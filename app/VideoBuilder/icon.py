"""The app icon: a dark tile with clips on a timeline (white) and the playhead in the one light-blue accent (house style: flat, no
gradients). FirstCut's icon: option B from the icon choices, dark version (2026-11-05).
python icon.py <out.png> -> a 1024 px PNG; scripts/package_videobuilder.sh turns it into AppIcon.icns."""
import sys

from PIL import Image, ImageDraw

SS = 4                                                        # drawn 4x larger, then scaled down: smooth edges
S = 1024 * SS
PAD = 100 * SS                                                # macOS icon grid: the tile is ~824 px of the 1024 canvas
U = (S - 2 * PAD) / 100                                       # one unit of the 100-unit design


def box(x, y, w, h):
    return (PAD + x * U, PAD + y * U, PAD + (x + w) * U, PAD + (y + h) * U)


TILE, CLIP, DIM, ACCENT = (22, 25, 29, 255), (242, 244, 246, 255), (120, 127, 134, 255), (90, 169, 230, 255)
im = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(im)
d.rounded_rectangle(box(0, 0, 100, 100), radius=22 * U, fill=TILE)
for x, y, w, fill in ((18, 29, 38, CLIP), (34, 45, 46, CLIP), (18, 61, 26, CLIP), (48, 61, 22, DIM)):   # three tracks of clips
    d.rounded_rectangle(box(x, y, w, 11), radius=3 * U, fill=fill)
lw = 3.5 * U                                                  # the playhead, over the clips
d.rounded_rectangle((PAD + 62 * U - lw / 2, PAD + 20 * U, PAD + 62 * U + lw / 2, PAD + 82 * U), radius=lw / 2, fill=ACCENT)
d.polygon([(PAD + 56 * U, PAD + 17 * U), (PAD + 68 * U, PAD + 17 * U), (PAD + 62 * U, PAD + 24 * U)], fill=ACCENT)
im.resize((1024, 1024), Image.LANCZOS).save(sys.argv[1] if len(sys.argv) > 1 else "icon.png")
