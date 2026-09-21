"""Articulated cutout rig built from the polished Mia sprite components."""

import math
from collections import deque
from pathlib import Path


class PolishedSpriteRig:
    PARTS = (
        "head", "torso", "left_upper_arm", "right_upper_arm",
        "left_forearm", "right_forearm", "left_thigh", "right_thigh",
        "left_lower_leg", "right_lower_leg",
    )

    def __init__(self, asset_dir: Path, width: int, height: int) -> None:
        from PIL import Image

        self.asset_dir = asset_dir
        self.width = width
        self.height = height
        self.parts = {
            name: self._largest_component(
                Image.open(asset_dir / f"rig_{name}.png").convert("RGBA")
            )
            for name in self.PARTS
        }

    @staticmethod
    def _largest_component(image):
        """Discard disconnected fragments introduced by component-sheet cell edges."""
        from PIL import Image

        alpha = image.getchannel("A")
        width, height = image.size
        solid = bytearray(
            1 if value > 24 else 0 for value in alpha.get_flattened_data()
        )
        visited = bytearray(width * height)
        largest: list[int] = []
        for start, value in enumerate(solid):
            if not value or visited[start]:
                continue
            visited[start] = 1
            queue = deque([start])
            component: list[int] = []
            while queue:
                index = queue.popleft()
                component.append(index)
                x, y = index % width, index // width
                for adjacent in (index - 1, index + 1, index - width, index + width):
                    if adjacent < 0 or adjacent >= width * height or visited[adjacent]:
                        continue
                    ax, ay = adjacent % width, adjacent // width
                    if abs(ax - x) + abs(ay - y) != 1 or not solid[adjacent]:
                        continue
                    visited[adjacent] = 1
                    queue.append(adjacent)
            if len(component) > len(largest):
                largest = component
        if not largest:
            return image
        keep = bytearray(width * height)
        for index in largest:
            keep[index] = 255
        mask = Image.frombytes("L", (width, height), bytes(keep))
        cleaned = Image.new("RGBA", image.size, (0, 0, 0, 0))
        cleaned.paste(image, mask=mask)
        box = cleaned.getbbox()
        return cleaned.crop(box) if box else image

    @classmethod
    def available(cls, asset_dir: Path) -> bool:
        return all((asset_dir / f"rig_{name}.png").is_file() for name in cls.PARTS)

    @staticmethod
    def _ease(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3 - 2 * value)

    @staticmethod
    def _joint(origin, length, angle):
        return origin[0] + length * math.sin(angle), origin[1] + length * math.cos(angle)

    @staticmethod
    def _fit_height(image, height):
        from PIL import Image

        width = max(1, round(image.width * height / image.height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def _fit_width(image, width):
        from PIL import Image

        height = max(1, round(image.height * width / image.width))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def _paste_center(canvas, image, center):
        canvas.alpha_composite(
            image, (round(center[0] - image.width / 2), round(center[1] - image.height / 2))
        )

    @staticmethod
    def _paste_jointed(canvas, image, joint, angle, pivot=(0.5, 0.10)):
        """Rotate one part about its normalized top joint and attach it to a world joint."""
        from PIL import Image

        pad = max(image.width, image.height) * 3
        work = Image.new("RGBA", (pad, pad), (0, 0, 0, 0))
        center = (pad // 2, pad // 2)
        px, py = image.width * pivot[0], image.height * pivot[1]
        work.alpha_composite(image, (round(center[0] - px), round(center[1] - py)))
        rotated = work.rotate(-math.degrees(angle), resample=Image.Resampling.BICUBIC,
                              center=center)
        canvas.alpha_composite(rotated, (round(joint[0] - center[0]),
                                         round(joint[1] - center[1])))

    def render(self, scene: int, progress: float):
        from PIL import Image

        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        body_x = self.width * 0.50
        base_y = self.height * 0.87
        crouch = jump = landing = 0.0
        if scene == 2:
            if progress < 0.18:
                crouch = self._ease(progress / 0.18)
            elif progress < 0.74:
                jump = math.sin((progress - 0.18) / 0.56 * math.pi)
            else:
                landing = math.sin((progress - 0.74) / 0.26 * math.pi)
        vertical = 15 * crouch - 74 * jump + 12 * landing

        torso = self._fit_width(self.parts["torso"], 104)
        head = self._fit_width(self.parts["head"], 118)
        upper = 55
        fore = 57
        thigh_len = 70
        lower_len = 80
        torso_center = (body_x, self.height * 0.55 + vertical)
        shoulder_y = torso_center[1] - 36
        hip_y = torso_center[1] + 47
        left_shoulder, right_shoulder = (body_x - 40, shoulder_y), (body_x + 40, shoulder_y)
        left_hip, right_hip = (body_x - 22, hip_y), (body_x + 22, hip_y)

        if scene == 1:
            left_arm = -0.10
            # Wave beside the body toward the rainy window without crossing the face.
            right_arm = 1.18 + 0.08 * math.sin(progress * math.tau * 2)
            left_fore = -0.02
            right_fore = -0.30 + 0.20 * math.sin(progress * math.tau * 2)
        elif scene == 2:
            spread = 0.72 + 0.18 * jump
            left_arm, right_arm = -spread, spread
            left_fore, right_fore = -0.20, 0.20
        elif scene == 4:
            raised = self._ease(progress / 0.46)
            left_arm, right_arm = -0.10, 0.35 + 1.02 * raised
            left_fore, right_fore = -0.02, 0.02
        else:
            left_arm, right_arm = -0.10, 0.10
            left_fore = right_fore = 0.0

        bend = 0.62 * crouch + 0.48 * landing + 0.25 * jump
        left_thigh_angle, right_thigh_angle = -0.10 - bend * 0.35, 0.10 + bend * 0.35
        left_lower_angle, right_lower_angle = 0.06 + bend * 0.62, -0.06 - bend * 0.62

        left_elbow = self._joint(left_shoulder, upper, left_arm)
        right_elbow = self._joint(right_shoulder, upper, right_arm)
        left_knee = self._joint(left_hip, thigh_len, left_thigh_angle)
        right_knee = self._joint(right_hip, thigh_len, right_thigh_angle)

        # Size each original painted component to the shared skeleton.
        parts = {
            "lua": self._fit_height(self.parts["left_upper_arm"], upper + 22),
            "rua": self._fit_height(self.parts["right_upper_arm"], upper + 22),
            "lfa": self._fit_height(self.parts["left_forearm"], fore + 30),
            "rfa": self._fit_height(self.parts["right_forearm"], fore + 30),
            "lt": self._fit_height(self.parts["left_thigh"], thigh_len + 25),
            "rt": self._fit_height(self.parts["right_thigh"], thigh_len + 25),
            "ll": self._fit_height(self.parts["left_lower_leg"], lower_len + 34),
            "rl": self._fit_height(self.parts["right_lower_leg"], lower_len + 34),
        }

        # Legs and arms sit behind the jacket; rounded overlaps hide rotation pivots.
        self._paste_jointed(canvas, parts["lt"], left_hip, left_thigh_angle)
        self._paste_jointed(canvas, parts["rt"], right_hip, right_thigh_angle)
        self._paste_jointed(canvas, parts["ll"], left_knee, left_lower_angle)
        self._paste_jointed(canvas, parts["rl"], right_knee, right_lower_angle)
        self._paste_jointed(canvas, parts["lua"], left_shoulder, left_arm)
        self._paste_jointed(canvas, parts["rua"], right_shoulder, right_arm)
        self._paste_center(canvas, torso, torso_center)
        self._paste_center(canvas, head, (body_x, torso_center[1] - 92))
        # Forearms and hands stay visible when a hand passes in front of the jacket
        # or face. Upper arms remain behind the torso to hide their shoulder seams.
        self._paste_jointed(canvas, parts["lfa"], left_elbow, left_arm + left_fore)
        self._paste_jointed(canvas, parts["rfa"], right_elbow, right_arm + right_fore)
        return canvas
