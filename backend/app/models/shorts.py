"""Validated contract for a four-scene, 15-second educational short."""

from pydantic import BaseModel, Field, field_validator, model_validator

MIN_SCENE_NARRATION_WORDS = 6
MAX_SCENE_NARRATION_WORDS = 9
MAX_TOTAL_NARRATION_WORDS = 36


class CharacterSpec(BaseModel):
    name: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    visual_identity: str = Field(min_length=1)
    age: int | str | None = None
    hair: str | None = None
    eyes: str | None = None
    clothes: str | None = None
    outfits: dict[str, str] = Field(default_factory=dict)
    states: dict[str, str] = Field(default_factory=dict)
    inferred_fields: list[str] = Field(default_factory=list)

    @field_validator("name", "kind", "visual_identity")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def add_default_outfit(self) -> "CharacterSpec":
        if not self.outfits:
            self.outfits = {"default": self.clothes or "unchanged from the reference"}
        self.outfits = {key.strip(): value.strip() for key, value in self.outfits.items()
                        if key.strip() and value.strip()}
        if not self.outfits:
            raise ValueError("character must define at least one usable outfit")
        if not self.states:
            self.states = {"default": self.visual_identity}
        self.states = {key.strip(): value.strip() for key, value in self.states.items()
                       if key.strip() and value.strip()}
        if not self.states:
            raise ValueError("character must define at least one usable state")
        return self


class StyleSpec(BaseModel):
    type: str = "2D children's book illustration"
    palette: str = "soft pastel"
    shapes: str = "rounded"
    lighting: str = "soft"
    outline: str = "clean"


class ShortSceneDraft(BaseModel):
    action: str = Field(min_length=1)
    location: str = Field(min_length=1)
    weather: str = Field(min_length=1)
    camera: str = Field(min_length=1)
    motion: str = Field(min_length=1)
    narration: str = Field(min_length=1)
    outfit_id: str = "default"
    state_id: str = "default"

    @field_validator("action", "location", "weather", "camera", "motion", "narration",
                     "outfit_id", "state_id")
    @classmethod
    def strip_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    @field_validator("narration")
    @classmethod
    def enforce_narration_budget(cls, value: str) -> str:
        words = len(value.split())
        if not MIN_SCENE_NARRATION_WORDS <= words <= MAX_SCENE_NARRATION_WORDS:
            raise ValueError(
                f"narration is {words} words, expected "
                f"{MIN_SCENE_NARRATION_WORDS}-{MAX_SCENE_NARRATION_WORDS}"
            )
        return value


class ShortPlanDraft(BaseModel):
    summary: str = Field(min_length=1)
    character: CharacterSpec
    scenes: list[ShortSceneDraft] = Field(min_length=4, max_length=4)

    @field_validator("summary")
    @classmethod
    def validate_summary(cls, value: str) -> str:
        value = value.strip()
        words = len(value.split())
        if not 8 <= words <= 30:
            raise ValueError(f"summary is {words} words, expected 8-30")
        return value

    @model_validator(mode="after")
    def validate_outfit_references(self) -> "ShortPlanDraft":
        unknown = {scene.outfit_id for scene in self.scenes} - set(self.character.outfits)
        if unknown:
            raise ValueError(f"scene outfit_id values are undefined: {sorted(unknown)}")
        unknown_states = {scene.state_id for scene in self.scenes} - set(self.character.states)
        if unknown_states:
            raise ValueError(f"scene state_id values are undefined: {sorted(unknown_states)}")
        return self


class ShortScene(ShortSceneDraft):
    scene: int = Field(ge=1, le=4)
    source_sentence_ids: list[int] = Field(min_length=1)
    text: str = Field(min_length=1)
    pages: list[int] = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)
    visual_prompt: str | None = None
    keyframe_prompt: str | None = None
    narration_seconds: float = Field(gt=0)


class ShortPlan(BaseModel):
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    character: CharacterSpec
    style: StyleSpec = Field(default_factory=StyleSpec)
    scenes: list[ShortScene] = Field(min_length=4, max_length=4)
    source_sentences: list[dict]
    duration_seconds: int = 15

    @model_validator(mode="after")
    def validate_plan(self) -> "ShortPlan":
        expected = [item["id"] for item in self.source_sentences]
        actual = [sentence_id for scene in self.scenes
                  for sentence_id in scene.source_sentence_ids]
        if [scene.scene for scene in self.scenes] != [1, 2, 3, 4]:
            raise ValueError("short plan needs four scenes in order")
        if actual != expected:
            raise ValueError("short scenes must cover each source sentence exactly once")
        if any(scene.outfit_id not in self.character.outfits for scene in self.scenes):
            raise ValueError("scene outfit_id must reference a global character outfit")
        if any(scene.state_id not in self.character.states for scene in self.scenes):
            raise ValueError("scene state_id must reference a global character state")
        narration_words = sum(len(scene.narration.split()) for scene in self.scenes)
        if narration_words > MAX_TOTAL_NARRATION_WORDS:
            raise ValueError("total narration exceeds the 15-second word budget")
        if self.duration_seconds != 15:
            raise ValueError("short video duration must be 15 seconds")
        return self
