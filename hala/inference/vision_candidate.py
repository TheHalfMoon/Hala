"""Experimental v2 image advice. CPU fakes and remote engines stay unqualified."""
import base64
import binascii
import hashlib
import io
import re
import time

from hala.contracts import Request, ContractError
from hala.contracts.proposal import Action, validate_action, validate_target
from hala.contracts.validation import object_fields, require, number, sequence, contains
from hala.inference import ServiceError

MAX_PNG_BYTES = 524288
MAX_PIXELS = 1048576
PROMPT_REVISION = 'h16-v2-normalized-point-v1'
INSTRUCTION = ('Return only JSON: {"point":[x,y]}. Use integer normalized coordinates '
               'from 0 to 1000 on each image axis. The supplied image is the entire '
               'coordinate frame. Do not execute actions or claim completion.')
BINDINGS = ('request_id', 'tenant_id', 'task_id', 'device_id', 'session_id', 'frame_id')


def bindings(request):
    return dict(request_id=request.request_id, tenant_id=request.tenant_id,
                task_id=request.task.task_id, device_id=request.device.device_id,
                session_id=request.device.session_id, frame_id=request.observation.generation)


def decode_point(raw, width, height):
    """Independent v2 decoder: one exact JSON object, optionally one json fence."""
    from hala.serving import strict_json
    require(type(raw) is str and 0 < len(raw.encode('utf-8')) <= 2048, 'Bounded model text required')
    raw = raw.strip()
    if raw.startswith('```json\n') and raw.endswith('\n```'):
        raw = raw[8:-4]
    try:
        value = strict_json(raw.encode('utf-8'))
    except ServiceError:
        raise ContractError('Invalid point JSON') from None
    object_fields(value, ('point',))
    coordinates = sequence(value['point'], 2, 2)
    for coordinate in coordinates:
        number(coordinate, 0, 1000, integer=True)
    return tuple(min(size-1, coordinate*size//1000)
                 for coordinate, size in zip(coordinates, (width, height)))


def parse_request(raw, tenant):
    object_fields(raw, ('schema', 'policy_request', 'media'))
    require(raw['schema'] == 'hala.image_request.v2', 'Unsupported candidate schema')
    request = Request.from_dict(raw['policy_request'])
    if request.tenant_id != tenant:
        raise ServiceError('tenant_mismatch')
    if request.cancelled:
        raise ServiceError('cancelled')
    require(not request.budget_exhausted, 'Budget exhausted')
    require('click' in request.allowed_actions and request.method_ceiling == 'WINDOW_TARGETED_INPUT', 'Click not permitted')
    frame = request.observation.image
    require(frame.format == 'png' and frame.scale == 1, 'Only PNG identity scaling supported')
    require(frame.width * frame.height <= MAX_PIXELS, 'Pixel budget exceeded')
    media = object_fields(raw['media'], ('png_base64', 'sha256', *BINDINGS, 'origin', 'dpi'))
    require(type(media['origin']) is list and media['origin'] == [0, 0] and all(type(v) is int for v in media['origin']), 'Origin ambiguity')
    require(type(media['dpi']) is int and media['dpi'] == 96, 'Unsupported DPI')
    require(all(media[key] == value for key, value in bindings(request).items()), 'Media binding mismatch')
    digest = media['sha256']
    require(type(digest) is str and re.fullmatch('[a-f0-9]{64}', digest), 'Invalid digest')
    encoded = media['png_base64']
    require(type(encoded) is str and len(encoded) <= 4*((MAX_PNG_BYTES+2)//3), 'Encoded image exceeds bound')
    try:
        png = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error):
        raise ContractError('Invalid base64') from None
    require(0 < len(png) <= MAX_PNG_BYTES and hashlib.sha256(png).hexdigest() == digest, 'Image digest mismatch')
    require(png.startswith(b'\x89PNG\r\n\x1a\n'), 'PNG required')
    from PIL import Image
    try:
        with Image.open(io.BytesIO(png)) as opened:
            require(opened.format == 'PNG' and opened.size == (frame.width, frame.height), 'Image size mismatch')
            require(not getattr(opened, 'is_animated', False), 'Animated images unsupported')
            opened.verify()
        with Image.open(io.BytesIO(png)) as opened:
            opened.load()
            image = opened.convert('RGB')
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ContractError('Invalid PNG') from None
    eligible = []
    for target in request.observation.accessibility:
        try:
            validate_target(target.element_ref, request)
            eligible.append(target)
        except ContractError:
            pass
    require(bool(eligible), 'No eligible observed targets')
    # Redaction declarations are trusted policy, but the actual media must also
    # have black pixels in those regions before any engine sees it.
    import math
    for x, y, width, height in frame.redactions:
        box = (math.floor(x), math.floor(y), math.ceil(x+width), math.ceil(y+height))
        require(all(high == 0 for low, high in image.crop(box).getextrema()), 'Redaction pixels are not black')
    return request, image, tuple(eligible), digest


class CandidateBackend:
    """Explicit server-owned experimental seam; never changes production readiness."""
    def __init__(self, engine=None, *, require_explicit_label=False):
        if type(require_explicit_label) is not bool:
            raise ValueError('require_explicit_label must be a boolean')
        self.engine = engine
        # Explicit opt-in only: unqualified research seam, never host authority.
        self.require_explicit_label = require_explicit_label

    def metadata(self):
        return {'name': 'vision_candidate_v2', 'ready': False, 'qualified': False,
                'candidate_available': self.engine is not None, 'capabilities': ['action'] if self.engine else []}

    def infer(self, request, image, eligible, digest, context):
        context.check()
        if self.engine is None:
            raise ServiceError('backend_unqualified')
        started = time.monotonic()
        # Private IDs/bounds stay server-side. The engine grounds only a goal.
        engine_request = {'goal': request.task.goal}
        output = self.engine(image, engine_request, context)
        context.check()
        try:
            point = decode_point(output, image.width, image.height)
            matches = [target for target in eligible if contains(target.bounds, point)]
            # An excluded target may still geometrically cover the proposed
            # point (e.g., its OTHER corner intersects a redaction). Reject
            # ambiguity against the complete observed accessibility set, not
            # just the eligible subset. No excluded/protected element can
            # silently disappear from the uniqueness check.
            observed_matches = [target for target in request.observation.accessibility
                                if contains(target.bounds, point)]
            require(len(matches) == 1 and len(observed_matches) == 1
                    and observed_matches[0].element_ref == matches[0].element_ref,
                    'Point must uniquely identify one eligible observed target')
            if self.require_explicit_label:
                from hala.inference.explicit_label_guard import unique_exact_label_match
                all_names = [target.name for target in request.observation.accessibility]
                require(unique_exact_label_match(request.task.goal, matches[0].name, all_names),
                        'Selected target label must uniquely match explicit user intent')
            action = Action(type='click', point=point,
                            element_ref_hint=matches[0].element_ref,
                            coordinate_space=request.observation.image.coordinate_space, image_scale=1)
            validate_action(action, request)
            provenance = dict(self.engine.provenance)
            object_fields(provenance, ('engine', 'model_id', 'model_revision', 'processor_revision', 'precision', 'prompt_revision'))
            from hala.contracts.validation import string
            for value in provenance.values():
                string(value, 256)
            provenance.update(qualified=False, latency_ms=(time.monotonic()-started)*1000,
                              serving_config_revision='h16-v2-bounded-v1')
            return {'schema': 'hala.click_proposal.v2', **bindings(request),
                    'media_sha256': digest, 'actions': [action.to_dict()],
                    'confidence': None, 'status': 'continue',
                    'execution_authorized': False, 'verification_required': True,
                    'provenance': provenance}
        except (ContractError, ValueError, TypeError, AttributeError, KeyError):
            raise ServiceError('invalid_backend_output') from None
