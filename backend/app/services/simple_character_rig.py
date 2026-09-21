"""Deterministic articulated 2D child rig for identity-safe animation."""

import json
import math
from pathlib import Path


DEFAULT_RIG = {
    "skin": "#E9A06F",
    "hair": "#55321F",
    "eyes": "#4A2B1A",
    "shirt": "#F3C62F",
    "pants": "#3F79B8",
    "boots": "#D8443E",
    "outline": "#35251F",
    "clip": "#FFD735",
}


class SimpleCharacterRig:
    """Render the same rounded cartoon child from reusable joint geometry."""

    def __init__(self, config_path: Path, width: int, height: int) -> None:
        self.config_path = config_path
        self.width = width
        self.height = height
        self.colors = dict(DEFAULT_RIG)
        if config_path.is_file():
            self.colors.update(json.loads(config_path.read_text(encoding="utf-8")))

    @staticmethod
    def config_for_keyframe(keyframe: Path) -> Path:
        return keyframe.parent / "simple_character.json"

    @classmethod
    def available(cls, keyframe: Path) -> bool:
        return cls.config_for_keyframe(keyframe).is_file()

    @staticmethod
    def _point(origin, length, angle):
        return (
            origin[0] + length * math.sin(angle),
            origin[1] + length * math.cos(angle),
        )

    def render(self, scene: int, progress: float):
        from PIL import Image, ImageDraw

        scale = 3
        width, height = self.width * scale, self.height * scale
        image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        c = self.colors

        def xy(point):
            return tuple(round(value * scale) for value in point)

        def ellipse(box, fill, outline=c["outline"], stroke=2.2):
            draw.ellipse(tuple(round(value * scale) for value in box), fill=fill,
                         outline=outline, width=max(1, round(stroke * scale)))

        def line(points, fill, width_px, joint=True):
            scaled = [xy(point) for point in points]
            draw.line(scaled, fill=fill, width=round(width_px * scale), joint="curve")
            if joint:
                radius = width_px * 0.50
                for point in points:
                    ellipse((point[0] - radius, point[1] - radius,
                             point[0] + radius, point[1] + radius), fill, fill, 0)

        # Timing curves describe actions rather than perpetual floating.
        ease = lambda value: max(0.0, min(1.0, value)) ** 2 * (
            3 - 2 * max(0.0, min(1.0, value))
        )
        body_x, ground = self.width * 0.50, self.height * 0.89
        crouch = jump = landing = 0.0
        if scene == 2:
            if progress < 0.18:
                crouch = ease(progress / 0.18)
            elif progress < 0.74:
                phase = (progress - 0.18) / 0.56
                jump = math.sin(phase * math.pi)
            else:
                landing = math.sin((progress - 0.74) / 0.26 * math.pi)
        body_y = self.height * 0.57 + 20 * crouch - 78 * jump + 17 * landing

        shoulder_y = body_y - 34
        hip_y = body_y + 50
        head_y = body_y - 105 + 3 * crouch
        shoulder_left = (body_x - 31, shoulder_y)
        shoulder_right = (body_x + 31, shoulder_y)
        hip_left = (body_x - 19, hip_y)
        hip_right = (body_x + 19, hip_y)

        left_upper = 42
        left_lower = 38
        right_upper = 42
        right_lower = 38
        if scene == 1:
            left_arm = (-0.08, -0.02)
            right_arm = (1.10 + 0.18 * math.sin(progress * math.tau * 2), 0.16)
        elif scene == 2:
            spread = 0.75 + 0.20 * jump
            left_arm = (-spread, -0.30)
            right_arm = (spread, 0.30)
        elif scene == 4:
            raise_amount = ease(progress / 0.42)
            left_arm = (-0.08, -0.03)
            right_arm = (0.45 + 0.72 * raise_amount, 0.08)
        else:
            left_arm = (-0.08, -0.02)
            right_arm = (0.08, 0.02)

        left_elbow = self._point(shoulder_left, left_upper, left_arm[0])
        left_hand = self._point(left_elbow, left_lower, left_arm[0] + left_arm[1])
        right_elbow = self._point(shoulder_right, right_upper, right_arm[0])
        right_hand = self._point(right_elbow, right_lower, right_arm[0] + right_arm[1])

        bend = 0.70 * crouch + 0.50 * landing + 0.42 * jump
        left_knee = (hip_left[0] - 11 * bend, hip_left[1] + 54 - 18 * bend)
        right_knee = (hip_right[0] + 11 * bend, hip_right[1] + 54 - 18 * bend)
        left_foot = (left_knee[0] - 8 * bend, ground - 7 - 78 * jump)
        right_foot = (right_knee[0] + 8 * bend, ground - 7 - 78 * jump)

        # Hair behind the face and body.
        ellipse((body_x - 48, head_y - 43, body_x + 48, head_y + 53), c["hair"])
        # Limbs use thick rounded strokes and overlapping joints, preventing seams.
        line((hip_left, left_knee, left_foot), c["pants"], 24)
        line((hip_right, right_knee, right_foot), c["pants"], 24)

        # Rounded torso.
        torso = (body_x - 38, body_y - 45, body_x + 38, body_y + 54)
        draw.rounded_rectangle(tuple(round(value * scale) for value in torso),
                               radius=18 * scale, fill=c["shirt"], outline=c["outline"],
                               width=round(2.2 * scale))
        line((shoulder_left, left_elbow), c["shirt"], 20)
        line((left_elbow, left_hand), c["skin"], 15)
        line((shoulder_right, right_elbow), c["shirt"], 20)
        line((right_elbow, right_hand), c["skin"], 15)
        # Boots remain attached to the animated legs.
        for foot in (left_foot, right_foot):
            draw.rounded_rectangle((round((foot[0] - 16) * scale), round((foot[1] - 10) * scale),
                                    round((foot[0] + 22) * scale), round((foot[1] + 12) * scale)),
                                   radius=8 * scale, fill=c["boots"], outline=c["outline"],
                                   width=round(2.2 * scale))

        # Face geometry is reused exactly in every scene.
        ellipse((body_x - 39, head_y - 38, body_x + 39, head_y + 40), c["skin"])
        draw.pieslice((round((body_x - 40) * scale), round((head_y - 42) * scale),
                       round((body_x + 40) * scale), round((head_y + 18) * scale)),
                      180, 360, fill=c["hair"])
        eye_y = head_y - 3 - (3 if scene == 3 else 0)
        blink = scene == 1 and 0.43 < (progress % 0.55) < 0.48
        if blink:
            line(((body_x - 19, eye_y), (body_x - 7, eye_y)), c["outline"], 2, False)
            line(((body_x + 7, eye_y), (body_x + 19, eye_y)), c["outline"], 2, False)
        else:
            ellipse((body_x - 22, eye_y - 8, body_x - 7, eye_y + 8), "white")
            ellipse((body_x + 7, eye_y - 8, body_x + 22, eye_y + 8), "white")
            ellipse((body_x - 17, eye_y - 5, body_x - 10, eye_y + 5), c["eyes"], c["eyes"], 0)
            ellipse((body_x + 12, eye_y - 5, body_x + 19, eye_y + 5), c["eyes"], c["eyes"], 0)
        # Nose, smile, and fixed hair clip.
        line(((body_x, head_y + 5), (body_x - 2, head_y + 11)), c["outline"], 1.4, False)
        draw.arc((round((body_x - 16) * scale), round((head_y + 10) * scale),
                  round((body_x + 16) * scale), round((head_y + 29) * scale)),
                 10, 170, fill=c["outline"], width=round(2 * scale))
        draw.rounded_rectangle((round((body_x - 34) * scale), round((head_y - 22) * scale),
                                round((body_x - 18) * scale), round((head_y - 14) * scale)),
                               radius=3 * scale, fill=c["clip"], outline=c["outline"],
                               width=round(1.3 * scale))

        return image.resize((self.width, self.height), Image.Resampling.LANCZOS)

    def save_reference(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.render(scene=3, progress=0.0).save(path)
