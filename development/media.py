"""Actual media qualification must occur on Linux with exact font and RAQM."""
import hashlib
import platform
from pathlib import Path

FONT_SHA256 = 'ae7b7855e115a5966d8b1b3f80f254ccc117ec86f9965e202ee2940453837280'


def render_frame(episode, state, output, font_path):
    from PIL import Image, ImageDraw, ImageFont, features, __version__
    if platform.system() != 'Linux' or not features.check_feature('raqm'):
        raise ValueError('qualified Linux RAQM required for all bilingual media')
    font_path = Path(font_path)
    if not font_path.is_file() or hashlib.sha256(font_path.read_bytes()).hexdigest() != FONT_SHA256:
        raise ValueError('pinned font required')
    if __version__ != '11.3.0':
        raise ValueError('pinned Pillow 11.3.0 required')
    if features.version_feature('raqm') != '0.10.1':
        raise ValueError('pinned RAQM 0.10.1 required')
    from development.simulator import layout, Simulator, WIDTH, HEIGHT
    sim = Simulator(episode)
    sim.state = dict(state)
    observation = sim.observation()
    image = Image.new('RGB', (WIDTH, HEIGHT), '#f5f5f5')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), 24, layout_engine=ImageFont.Layout.RAQM)
    ar = episode['language'] == 'ar'
    def text(x, y, value, right=720):
        draw.text((right if ar else x, y), value, font=font, fill='black',
                  anchor='ra' if ar else 'la', direction='rtl' if ar else 'ltr', language='ar' if ar else 'en')
    text(60, 20, episode['title'])
    boxes = layout(episode)
    fx, fy, fw, fh = boxes['field']
    text(fx, fy-35, episode['field_label'], fx+fw)
    ix, iy, iw, ih = boxes['item']
    text(ix, iy-35, episode['source'], ix+iw)
    for name, box in boxes.items():
        x, y, w, h = box
        draw.rectangle((x, y, x+w-1, y+h-1), fill='white', outline='#333333', width=3)
        label = {'field': state['value'], 'item': episode['item'] if state['location'] == episode['source'] else '',
                 'destination': episode['destination'], 'save': observation['save_label']}[name]
        text(x+10, y+15, label, x+w-10)
    if state['location'] == episode['destination']:
        x, y, w, h = boxes['destination']
        text(x+10, y+85, episode['item'], x+w-10)
    # Keep diagnostic status short enough to fit; full visible observation also
    # accompanies the image in the prompt.
    status = ('فشل مؤقت' if ar else 'Temporary failure') if state['status'] == 'transient_failure' else observation['status']
    text(70, 560, status)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format='PNG')
    return {'sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'width': WIDTH, 'height': HEIGHT,
            'font_sha256': FONT_SHA256, 'pillow': __version__, 'raqm': features.version_feature('raqm'),
            'platform': platform.system(), 'language': episode['language']}
