"""ROS/model-independent selection for the YOLO-only test path.

Same-class continuity is not an instance tracking ID. Use one item for initial
bench tests. The legacy two-stage classifier module is deliberately untouched.
"""
from dataclasses import dataclass
import math

# Must match my_yolo_cpp_pkg/classify_yolo_information.py object_id and
# navigation/auto_nav.py object_name. glass_bottle is out of scope.
CLASS_IDS = {'can': 0, 'paper': 1, 'plastic': 2,
             'trash': 3, 'person': 4}


def build_model_to_project_class_map(model_names, required_project_ids=(0, 1, 2)):
    """Validate required class names once and map model IDs to project IDs.

    Model class order may change and additional classes are allowed. The project
    contract is the stable class *name* (e.g. can/paper/plastic), while downstream
    ROS messages keep their existing project IDs. Unknown extra model classes are
    intentionally ignored.
    """
    required_ids = tuple(int(i) for i in required_project_ids)
    if not required_ids or any(i not in CLASS_IDS.values() for i in required_ids):
        raise ValueError(f'Invalid required project class IDs: {required_ids}')

    project_name_by_id = {project_id: name for name, project_id in CLASS_IDS.items()}
    required_names = {project_name_by_id[i]: i for i in required_ids}

    if isinstance(model_names, dict):
        raw_items = model_names.items()
    else:
        raw_items = enumerate(model_names)

    model_to_project = {}
    matched_model_ids = {}
    normalized_model_names = {}
    for raw_model_id, raw_name in raw_items:
        try:
            model_id = int(raw_model_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid model class ID: {raw_model_id!r}') from exc
        name = str(raw_name).strip().lower()
        normalized_model_names[model_id] = name
        if name not in required_names:
            continue  # Extra model classes are allowed but not used by this project.
        project_id = required_names[name]
        if project_id in matched_model_ids:
            raise ValueError(
                f'Ambiguous model classes for required name {name!r}: '
                f'model IDs {matched_model_ids[project_id]} and {model_id}'
            )
        matched_model_ids[project_id] = model_id
        model_to_project[model_id] = project_id

    missing = [project_name_by_id[i] for i in required_ids if i not in matched_model_ids]
    if missing:
        raise ValueError(
            f'Model is missing required project classes {missing}; '
            f'model classes={normalized_model_names}'
        )
    return model_to_project


@dataclass(frozen=True)
class Candidate:
    class_id: int
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def coord(self):
        return [self.x, self.y, self.width, self.height]

    def valid(self):
        return (self.class_id >= 0 and 0 < self.confidence <= 1
                and all(math.isfinite(v) for v in [self.confidence] + self.coord)
                and self.x >= 0 and self.y >= 0 and self.width > 0 and self.height > 0)


class StableSelector:
    def __init__(self, required_frames=2, reference_x=350.0,
                 allowed_ids=(0, 1, 2), target_class_id=-1):
        if required_frames < 1 or not math.isfinite(reference_x):
            raise ValueError('Invalid selector settings')
        self.required_frames = required_frames
        self.reference_x = reference_x
        self.allowed_ids = tuple(allowed_ids)
        self.target_class_id = target_class_id
        self.reset()

    def reset(self):
        self.last_class = None
        self.streak = 0
        # Once tracking has confirmed a class, keep that class lock across a
        # brief visual dropout. This is still class continuity, not instance ID.
        self.confirmed_tracking_class = None

    def select(self, candidates, tracking):
        usable = [c for c in candidates if c.valid() and c.class_id in self.allowed_ids
                  and (self.target_class_id < 0 or c.class_id == self.target_class_id)]
        if not usable:
            self.last_class = None
            self.streak = 0
            if not tracking:
                self.confirmed_tracking_class = None
            return None

        # During an active track, a class that was already confirmed does not
        # need the initial required_frames gate again after a short dropout.
        # Also avoid silently switching to another class while the locked class
        # is temporarily absent.
        if tracking and self.confirmed_tracking_class is not None:
            same_class = [c for c in usable if c.class_id == self.confirmed_tracking_class]
            if not same_class:
                self.last_class = None
                self.streak = 0
                return None
            best = min(same_class, key=lambda c: abs(c.x - self.reference_x))
            self.last_class = best.class_id
            self.streak = self.required_frames
            return best

        best = (min(usable, key=lambda c: abs(c.x - self.reference_x)) if tracking
                else max(usable, key=lambda c: c.confidence))
        self.streak = self.streak + 1 if self.last_class == best.class_id else 1
        self.last_class = best.class_id
        if self.streak < self.required_frames:
            return None
        if tracking:
            self.confirmed_tracking_class = best.class_id
        return best
