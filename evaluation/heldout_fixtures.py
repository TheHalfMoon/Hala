"""Original prospective H14 GUI specifications, rendering and freeze verification."""
import hashlib
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / 'evaluation/heldout/tasks.json'
MANIFEST = ROOT / 'evaluation/heldout/manifest.json'
FAMILIES = ('calendar_dates', 'transit_arrivals', 'warehouse_stock', 'kanban_priority',
            'seat_locations', 'chart_peaks', 'palette_colors', 'document_versions')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')


def author_cases():
    """Authoring only. Runtime loads frozen specifications and never regenerates them."""
    cases = []
    for fi, family in enumerate(FAMILIES):
        for variant in range(8):
            ar = variant >= 4
            rng = random.Random(2026101400 + fi * 101 + variant)
            width, height = ((800, 600), (960, 640), (1024, 768), (880, 720))[variant % 4]
            n = 6 if family not in ('calendar_dates', 'seat_locations') else 12
            target = rng.randrange(n)
            order = list(range(n))
            rng.shuffle(order)
            controls = []
            title = [('Appointment calendar', 'تقويم المواعيد'), ('Transit board', 'لوحة الرحلات'),
                     ('Stock levels', 'مستويات المخزون'), ('Work board', 'لوحة العمل'),
                     ('Seat map', 'خريطة المقاعد'), ('Monthly demand', 'الطلب الشهري'),
                     ('Color palette', 'لوحة الألوان'), ('Document history', 'سجل المستند')][fi][ar]
            values = rng.sample(range(11, 95), n)
            if family == 'calendar_dates':
                chosen = 1 + target
                goal = (f'Select day {chosen} to schedule the appointment.' if not ar else
                        f'اختر اليوم {chosen} لتحديد الموعد.')
            elif family == 'transit_arrivals':
                target = values.index(min(values))
                goal = ('Select the route with the earliest arrival time.' if not ar else
                        'اختر المسار الذي يصل في أبكر وقت.')
            elif family == 'warehouse_stock':
                target = values.index(min(values))
                goal = ('Select the item with the lowest stock count for replenishment.' if not ar else
                        'اختر الصنف الأقل مخزونا لإعادة التزويد.')
            elif family == 'kanban_priority':
                goal = ('Select the card marked Urgent in the Review column.' if not ar else
                        'اختر البطاقة العاجلة في عمود المراجعة.')
            elif family == 'seat_locations':
                chosen = f'{1 + target // 4 + variant * 3}{"ABCD"[target % 4]}'
                goal = (f'Reserve seat {chosen}.' if not ar else f'احجز المقعد {chosen}.')
            elif family == 'chart_peaks':
                target = values.index(max(values))
                goal = ('Select the bar for the month with the highest demand.' if not ar else
                        'اختر عمود الشهر الأعلى طلبا.')
            elif family == 'palette_colors':
                colors = [('Crimson', 'قرمزي', '#c62d42'), ('Teal', 'أزرق مخضر', '#168f92'),
                          ('Gold', 'ذهبي', '#dda520'), ('Violet', 'بنفسجي', '#9160c7'),
                          ('Olive', 'زيتوني', '#7c8c36'), ('Coral', 'مرجاني', '#ea806b')]
                name = colors[target][ar]
                goal = (f'Choose the {name} swatch for the heading.' if not ar else
                        f'اختر اللون {name} للعنوان.')
            else:
                target = values.index(max(values))
                goal = ('Select the latest version of the document.' if not ar else
                        'اختر أحدث إصدار من المستند.')
            for slot, identity in enumerate(order):
                # Geometry differs by family; jitter and canvas dimensions differ by variant.
                if family == 'calendar_dates':
                    cols, x0, y0, cw, ch = 4, 55, 155, (width - 110) // 4, 100
                elif family == 'seat_locations':
                    cols, x0, y0, cw, ch = 4, 125, 160, (width - 250) // 4, 100
                elif family == 'kanban_priority':
                    cols, x0, y0, cw, ch = 3, 30, 175, (width - 60) // 3, 155
                elif family == 'palette_colors':
                    cols, x0, y0, cw, ch = 3, 55, 145, (width - 110) // 3, 165
                elif family == 'chart_peaks':
                    cols, x0, y0, cw, ch = 6, 45, 150, (width - 90) // 6, height - 220
                elif family == 'warehouse_stock':
                    cols, x0, y0, cw, ch = 2, 45, 130, (width - 90) // 2, 125
                else:
                    cols, x0, y0, cw, ch = 1, 75, 125, width - 150, (height - 160) // 6
                x = x0 + (slot % cols) * cw + rng.randrange(0, 9)
                y = y0 + (slot // cols) * ch + rng.randrange(0, 7)
                box = [x, y, x + cw - 20, y + ch - 15]
                fill = '#ffffff'
                if family == 'calendar_dates':
                    label = str(1 + identity)
                elif family == 'transit_arrivals':
                    # Unique times aligned with the private earliest-time label.
                    minutes = values[identity]
                    label = (f'Route {identity + 1}   09:{minutes % 60:02d}' if not ar else
                             f'مسار {identity + 1}   09:{minutes % 60:02d}')
                    # Hour rollover prevents modulo ordering ambiguity.
                    label = label.replace('09:', f'{9 + minutes // 60:02d}:')
                elif family == 'warehouse_stock':
                    label = (f'Item {identity + 1}: {values[identity]} units' if not ar else
                             f'صنف {identity + 1}: {values[identity]} وحدة')
                elif family == 'kanban_priority':
                    # Put unique urgent review card into a designated review column;
                    # the same word occurs in another column as a distractor.
                    review_slot = order.index(target)
                    review_col = review_slot % 3
                    priority = ('Urgent' if not ar else 'عاجل') if (identity == target or
                        slot % 3 != review_col) else ('Routine' if not ar else 'عادي')
                    label = (f'Task {identity + 1}\n{priority}' if not ar else
                             f'مهمة {identity + 1}\n{priority}')
                elif family == 'seat_locations':
                    label = f'{1 + identity // 4 + variant * 3}{"ABCD"[identity % 4]}'
                elif family == 'chart_peaks':
                    label = (f'M{identity + 1}: {values[identity]}' if not ar else
                             f'ش{identity + 1}: {values[identity]}')
                    box[1] = height - 90 - values[identity] * (height - 230) // 100
                    box[3] = height - 90
                    fill = '#80b7d9'
                elif family == 'palette_colors':
                    label, fill = colors[identity][ar], colors[identity][2]
                else:
                    label = (f'Version {values[identity]} - draft {identity + 1}' if not ar else
                             f'إصدار {values[identity]} - مسودة {identity + 1}')
                controls.append({'id': f'item-{identity}', 'box': box, 'label': label, 'fill': fill})
            decorations = []
            if family == 'kanban_priority':
                review_col = order.index(target) % 3
                names = ['Backlog', 'Review', 'Done'] if not ar else ['انتظار', 'مراجعة', 'منجز']
                # Assign Review to the authored target column, no expected marker in the image.
                names[1], names[review_col] = names[review_col], names[1]
                decorations = [{'position': [30 + c * ((width - 60) // 3), 130], 'text': name}
                               for c, name in enumerate(names)]
            cases.append({'id': f'h14-{family}-{variant:02d}', 'family': family,
                          'language': 'ar' if ar else 'en', 'size': [width, height],
                          'dpi': [96, 144, 192, 120][variant % 4], 'title': title,
                          'goal': goal, 'controls': controls, 'decorations': decorations,
                          'expected_target': f'item-{target}', 'fixture': 'one_click_selection_v1'})
    return {'schema': 'hala.h14.heldout.v1', 'rights': {'source': 'original_authored',
            'license': 'CC0-1.0', 'personal_data': False, 'third_party_assets': False},
            'training_allowed': False, 'cases': cases}


def validate(dataset):
    cases = dataset['cases']
    if len(cases) != 64 or len({c['id'] for c in cases}) != 64:
        raise ValueError('Exactly 64 unique prospective instances required')
    if set(c['family'] for c in cases) != set(FAMILIES):
        raise ValueError('Eight original families required')
    for family in FAMILIES:
        for language in ('en', 'ar'):
            if sum(c['family'] == family and c['language'] == language for c in cases) != 4:
                raise ValueError('Balanced family/language counts required')
    for case in cases:
        width, height = case['size']
        ids = [c['id'] for c in case['controls']]
        if len(set(ids)) != len(ids) or case['expected_target'] not in ids:
            raise ValueError('Unique controls and known private target required')
        for i, control in enumerate(case['controls']):
            l, t, r, b = control['box']
            if not 0 <= l < r <= width or not 0 <= t < b <= height:
                raise ValueError('Visible half-open boxes required')
            for other in case['controls'][i + 1:]:
                ol, ot, oright, ob = other['box']
                if max(l, ol) < min(r, oright) and max(t, ot) < min(b, ob):
                    raise ValueError('Overlapping controls are ambiguous')
    return dataset


def verify_freeze(expected_manifest_hash):
    if digest(MANIFEST) != expected_manifest_hash:
        raise ValueError('Manifest differs from trusted prior freeze')
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for relative, expected in manifest['files'].items():
        path = ROOT / relative
        if path.resolve().is_relative_to(ROOT) is False or digest(path) != expected:
            raise ValueError('Frozen source/specification mismatch')
    return validate(json.loads(FROZEN.read_text(encoding='utf-8')))


def render(case, font_path):
    from PIL import Image, ImageDraw, ImageFont, features
    if case['language'] == 'ar' and not features.check('raqm'):
        raise ValueError('Actual RAQM shaping required for Arabic rendering')
    font = ImageFont.truetype(str(font_path), 20, layout_engine=ImageFont.Layout.RAQM
                              if features.check('raqm') else ImageFont.Layout.BASIC)
    image = Image.new('RGB', tuple(case['size']), '#edf1f4')
    draw = ImageDraw.Draw(image)
    kwargs = {'direction': 'rtl', 'language': 'ar'} if case['language'] == 'ar' else {}
    draw.text((35, 35), case['title'], fill='#19222b', font=font, **kwargs)
    for decoration in case['decorations']:
        draw.text(tuple(decoration['position']), decoration['text'], fill='black', font=font, **kwargs)
    for control in case['controls']:
        l, t, r, b = control['box']
        draw.rectangle((l, t, r - 1, b - 1), fill=control['fill'], outline='#344455', width=2)
        # Fit all lines before rendering; fail rather than silently clipping Arabic or English.
        text = control['label']
        bounds = draw.multiline_textbbox((0, 0), text, font=font, **kwargs)
        if bounds[2] - bounds[0] > r - l - 8 or bounds[3] - bounds[1] > b - t - 8:
            raise ValueError('Authored label exceeds control bounds')
        draw.multiline_text(((l + r) / 2, (t + b) / 2), text, fill='#101010', font=font,
                            anchor='mm', align='center', **kwargs)
    return image
