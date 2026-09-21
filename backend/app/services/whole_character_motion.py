"""Animate cohesive full-body character poses without articulated cutout seams."""

import math
from pathlib import Path

from backend.app.services.sprite_character_rig import PolishedSpriteRig


class WholeCharacterPoseAnimator:
    """Render one complete painted character per frame from a 3x2 pose sheet."""

    SHEET_NAME = "full_pose_sheet.png"

    def __init__(self, asset_dir: Path, width: int, height: int) -> None:
        from PIL import Image

        self.width = width
        self.height = height
        sheet = Image.open(asset_dir / self.SHEET_NAME).convert("RGBA")
        cell_width, cell_height = sheet.width // 3, sheet.height // 2
        self.poses = []
        for row in range(2):
            for column in range(3):
                cell = sheet.crop((
                    column * cell_width,
                    row * cell_height,
                    (column + 1) * cell_width,
                    (row + 1) * cell_height,
                ))
                self.poses.append(PolishedSpriteRig._largest_component(cell))

    @classmethod
    def available(cls, asset_dir: Path) -> bool:
        return (asset_dir / cls.SHEET_NAME).is_file()

    @staticmethod
    def _ease(value: float) -> float:
        value = max(0.0, min(1.0, value))
        return value * value * (3 - 2 * value)

    @staticmethod
    def _fit_height(image, height: int):
        from PIL import Image

        width = max(1, round(image.width * height / image.height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    def render(self, scene: int, progress: float):
        from PIL import Image

        canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        pose_index = {1: 0, 3: 4, 4: 5}.get(scene, 4)
        lift = 0.0
        scale = 1.0

        if scene == 1:
            # The full figure stays grounded while the gesture gets a small body emphasis.
            scale = 1.0 + 0.008 * math.sin(progress * math.tau * 2)
        elif scene == 2:
            if progress < 0.20:
                pose_index = 1
                scale = 1.0 - 0.025 * self._ease(progress / 0.20)
            elif progress < 0.73:
                pose_index = 2
                flight = (progress - 0.20) / 0.53
                lift = 76 * math.sin(flight * math.pi)
                scale = 1.0 + 0.025 * math.sin(flight * math.pi)
            else:
                pose_index = 3
                impact = math.sin((progress - 0.73) / 0.27 * math.pi)
                scale = 1.0 - 0.035 * impact
        elif scene == 3:
            # A subtle breathing change, anchored at the boot soles.
            scale = 1.0 + 0.006 * math.sin(progress * math.tau)
        elif scene == 4:
            pose_index = 4 if progress < 0.22 else 5

        target_height = round(self.height * 0.70 * scale)
        figure = self._fit_height(self.poses[pose_index], target_height)
        x = round((self.width - figure.width) / 2)
        ground_y = self.height * (0.875 if scene == 2 else 0.89)
        y = round(ground_y - figure.height - lift)
        canvas.alpha_composite(figure, (x, y))
        return canvas
