from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from .rle import compact_rle_to_mask, mask_area, mask_bbox, mask_to_compact_rle, mask_to_png_data_url


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


@dataclass(frozen=True)
class ImageEntry:
    index: int
    file_name: str
    path: Path
    image_id: int | None
    width: int | None
    height: int | None
    in_coco: bool


class COCOStore:
    def __init__(self, coco_path: Path, images_dir: Path, output_path: Path):
        self.coco_path = coco_path
        self.images_dir = images_dir
        self._default_output_path = output_path
        self.output_path = output_path
        self.source_name = coco_path.name
        with self.coco_path.open("r", encoding="utf-8") as fh:
            source_data: dict[str, Any] = json.load(fh)
        self.open_data(source_data, self.coco_path.name, self._default_output_path)

    def reload(self) -> None:
        self._load_data(copy.deepcopy(self._source_data))

    def open_data(self, data: dict[str, Any], source_name: str, output_path: Path | None = None) -> dict[str, Any]:
        if not isinstance(data, dict):
            raise ValueError("annotation data must be a JSON object")
        self._validate_source_name(source_name)
        self._validate_coco_data(data)
        self.source_name = Path(source_name).name or self.coco_path.name
        self.output_path = output_path or self._output_path_for_source(self.source_name)
        self._source_data = copy.deepcopy(data)
        self._load_data(copy.deepcopy(data))
        return {
            "source_name": self.source_name,
            "output_path": str(self.output_path),
            "image_count": len(self.entries),
        }

    def _output_path_for_source(self, source_name: str) -> Path:
        stem = Path(source_name).stem or self.coco_path.stem
        return self._default_output_path.parent / f"{stem}.corrected.json"

    @staticmethod
    def _validate_source_name(source_name: str) -> None:
        if Path(source_name).suffix.lower() != ".json":
            raise ValueError("annotation file must have a .json extension")

    @staticmethod
    def _validate_coco_data(data: dict[str, Any]) -> None:
        for key in ("images", "annotations", "categories"):
            if key not in data:
                raise ValueError(f"annotation data is not COCO format: missing required key '{key}'")
            if not isinstance(data[key], list):
                raise ValueError(f"annotation data is not COCO format: '{key}' must be a list")

        image_ids: set[int] = set()
        for idx, image in enumerate(data["images"]):
            if not isinstance(image, dict):
                raise ValueError(f"annotation data is not COCO format: images[{idx}] must be an object")
            for key in ("id", "file_name"):
                if key not in image:
                    raise ValueError(f"annotation data is not COCO format: images[{idx}] missing '{key}'")
            image_id = COCOStore._int_field(image["id"], f"images[{idx}].id")
            if image_id in image_ids:
                raise ValueError(f"annotation data is not COCO format: duplicate image id {image_id}")
            image_ids.add(image_id)
            if not isinstance(image["file_name"], str) or not image["file_name"]:
                raise ValueError(f"annotation data is not COCO format: images[{idx}].file_name must be a non-empty string")

        category_ids: set[int] = set()
        for idx, category in enumerate(data["categories"]):
            if not isinstance(category, dict):
                raise ValueError(f"annotation data is not COCO format: categories[{idx}] must be an object")
            if "id" not in category:
                raise ValueError(f"annotation data is not COCO format: categories[{idx}] missing 'id'")
            category_id = COCOStore._int_field(category["id"], f"categories[{idx}].id")
            if category_id in category_ids:
                raise ValueError(f"annotation data is not COCO format: duplicate category id {category_id}")
            category_ids.add(category_id)

        annotation_ids: set[int] = set()
        for idx, annotation in enumerate(data["annotations"]):
            if not isinstance(annotation, dict):
                raise ValueError(f"annotation data is not COCO format: annotations[{idx}] must be an object")
            for key in ("id", "image_id", "category_id", "segmentation", "area", "bbox"):
                if key not in annotation:
                    raise ValueError(f"annotation data is not COCO format: annotations[{idx}] missing '{key}'")
            annotation_id = COCOStore._int_field(annotation["id"], f"annotations[{idx}].id")
            if annotation_id in annotation_ids:
                raise ValueError(f"annotation data is not COCO format: duplicate annotation id {annotation_id}")
            annotation_ids.add(annotation_id)
            image_id = COCOStore._int_field(annotation["image_id"], f"annotations[{idx}].image_id")
            if image_id not in image_ids:
                raise ValueError(f"annotation data is not COCO format: annotations[{idx}].image_id does not reference an image")
            category_id = COCOStore._int_field(annotation["category_id"], f"annotations[{idx}].category_id")
            if category_ids and category_id not in category_ids:
                raise ValueError(f"annotation data is not COCO format: annotations[{idx}].category_id does not reference a category")
            COCOStore._validate_rle(annotation["segmentation"], f"annotations[{idx}].segmentation")
            if not isinstance(annotation["bbox"], list) or len(annotation["bbox"]) != 4:
                raise ValueError(f"annotation data is not COCO format: annotations[{idx}].bbox must be a 4-value list")
            if not isinstance(annotation["area"], int | float):
                raise ValueError(f"annotation data is not COCO format: annotations[{idx}].area must be a number")

    @staticmethod
    def _int_field(value: Any, field_name: str) -> int:
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"annotation data is not COCO format: {field_name} must be an integer") from exc

    @staticmethod
    def _validate_rle(segmentation: Any, field_name: str) -> None:
        if not isinstance(segmentation, dict):
            raise ValueError(f"annotation data is not supported COCO RLE format: {field_name} must be an RLE object")
        if "size" not in segmentation or "counts" not in segmentation:
            raise ValueError(f"annotation data is not supported COCO RLE format: {field_name} must contain 'size' and 'counts'")
        size = segmentation["size"]
        counts = segmentation["counts"]
        if not isinstance(size, list) or len(size) != 2:
            raise ValueError(f"annotation data is not supported COCO RLE format: {field_name}.size must be a 2-value list")
        if not isinstance(counts, str | list):
            raise ValueError(f"annotation data is not supported COCO RLE format: {field_name}.counts must be a string or list")

    def _load_data(self, data: dict[str, Any]) -> None:
        self.data = data
        self.images: list[dict[str, Any]] = self.data.setdefault("images", [])
        self.annotations: list[dict[str, Any]] = self.data.setdefault("annotations", [])
        self.categories: list[dict[str, Any]] = self.data.setdefault("categories", [])
        self._original_annotation_ids = {int(ann["id"]) for ann in self.annotations}
        self._image_by_id = {int(image["id"]): image for image in self.images}
        self._image_by_file = {image["file_name"]: image for image in self.images}
        self._annotation_by_id = {int(ann["id"]): ann for ann in self.annotations}
        self._rebuild_annotation_index()
        self.entries = self._build_entries()

    def _rebuild_annotation_index(self) -> None:
        self._annotations_by_image_id: dict[int, list[dict[str, Any]]] = {}
        for ann in self.annotations:
            self._annotations_by_image_id.setdefault(int(ann["image_id"]), []).append(ann)

    def _build_entries(self) -> list[ImageEntry]:
        disk_files = {
            path.name: path
            for path in self.images_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        }
        entries: list[ImageEntry] = []
        seen: set[str] = set()
        for image in self.images:
            file_name = image["file_name"]
            path = disk_files.get(file_name, self.images_dir / file_name)
            entries.append(
                ImageEntry(
                    index=len(entries) + 1,
                    file_name=file_name,
                    path=path,
                    image_id=int(image["id"]),
                    width=int(image["width"]) if image.get("width") is not None else None,
                    height=int(image["height"]) if image.get("height") is not None else None,
                    in_coco=True,
                )
            )
            seen.add(file_name)
        for file_name in sorted(set(disk_files) - seen):
            entries.append(
                ImageEntry(
                    index=len(entries) + 1,
                    file_name=file_name,
                    path=disk_files[file_name],
                    image_id=None,
                    width=None,
                    height=None,
                    in_coco=False,
                )
            )
        return entries

    def list_images(self) -> list[dict[str, Any]]:
        return [
            {
                "index": entry.index,
                "file_name": entry.file_name,
                "image_id": entry.image_id,
                "width": entry.width,
                "height": entry.height,
                "in_coco": entry.in_coco,
                "annotation_count": len(self._annotations_by_image_id.get(entry.image_id or -1, [])),
            }
            for entry in self.entries
        ]

    def entry_at(self, index: int) -> ImageEntry:
        if index < 1 or index > len(self.entries):
            raise IndexError(f"image index {index} is out of range")
        return self.entries[index - 1]

    def image_path(self, index: int) -> Path:
        entry = self.entry_at(index)
        if not entry.path.exists():
            raise FileNotFoundError(entry.path)
        return entry.path

    def image_detail(self, index: int) -> dict[str, Any]:
        entry = self.entry_at(index)
        width, height = self._dimensions(entry)
        annotations = []
        if entry.image_id is not None:
            for ann in self._annotations_by_image_id.get(entry.image_id, []):
                mask = compact_rle_to_mask(ann["segmentation"])
                annotations.append(
                    {
                        "id": int(ann["id"]),
                        "category_id": int(ann["category_id"]),
                        "area": float(ann.get("area", 0.0)),
                        "bbox": ann.get("bbox", [0.0, 0.0, 0.0, 0.0]),
                        "is_new": int(ann["id"]) not in self._original_annotation_ids,
                        "mask_png": mask_to_png_data_url(mask),
                    }
                )
        return {
            "index": entry.index,
            "file_name": entry.file_name,
            "image_id": entry.image_id,
            "width": width,
            "height": height,
            "in_coco": entry.in_coco,
            "image_url": f"/api/images/{entry.index}/file",
            "annotations": annotations,
        }

    def create_instance(self, index: int, mask) -> dict[str, Any]:
        entry = self.entry_at(index)
        image = self._ensure_image_record(entry)
        ann_id = self._next_annotation_id()
        ann = {
            "id": ann_id,
            "image_id": int(image["id"]),
            "category_id": self._default_category_id(),
            "segmentation": mask_to_compact_rle(mask),
            "area": mask_area(mask),
            "bbox": mask_bbox(mask),
            "iscrowd": 0,
        }
        self.annotations.append(ann)
        self._annotation_by_id[ann_id] = ann
        self._rebuild_annotation_index()
        return self.annotation_response(ann)

    def update_mask(self, annotation_id: int, mask) -> dict[str, Any]:
        ann = self._annotation_by_id[annotation_id]
        ann["segmentation"] = mask_to_compact_rle(mask)
        ann["area"] = mask_area(mask)
        return self.annotation_response(ann)

    def delete_annotation(self, annotation_id: int) -> dict[str, Any]:
        ann = self._annotation_by_id.pop(annotation_id)
        self.annotations = [item for item in self.annotations if int(item["id"]) != annotation_id]
        self.data["annotations"] = self.annotations
        self._rebuild_annotation_index()
        return {"deleted_id": int(ann["id"]), "image_id": int(ann["image_id"])}

    def annotation_response(self, ann: dict[str, Any]) -> dict[str, Any]:
        mask = compact_rle_to_mask(ann["segmentation"])
        return {
            "id": int(ann["id"]),
            "image_id": int(ann["image_id"]),
            "category_id": int(ann["category_id"]),
            "area": float(ann.get("area", 0.0)),
            "bbox": ann.get("bbox", [0.0, 0.0, 0.0, 0.0]),
            "is_new": int(ann["id"]) not in self._original_annotation_ids,
            "mask_png": mask_to_png_data_url(mask),
        }

    def save(self) -> Path:
        out = copy.deepcopy(self.data)
        out["images"] = self.images
        out["annotations"] = self.annotations
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        return self.output_path

    def saved_json(self) -> dict[str, Any]:
        if self.output_path.exists():
            with self.output_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        return self.data

    def _dimensions(self, entry: ImageEntry) -> tuple[int, int]:
        if entry.width and entry.height:
            return entry.width, entry.height
        with Image.open(entry.path) as image:
            return image.size

    def _ensure_image_record(self, entry: ImageEntry) -> dict[str, Any]:
        if entry.image_id is not None:
            return self._image_by_id[entry.image_id]
        if entry.file_name in self._image_by_file:
            return self._image_by_file[entry.file_name]
        width, height = self._dimensions(entry)
        image_id = self._next_image_id()
        image = {"id": image_id, "file_name": entry.file_name, "width": width, "height": height}
        self.images.append(image)
        self._image_by_id[image_id] = image
        self._image_by_file[entry.file_name] = image
        updated = ImageEntry(entry.index, entry.file_name, entry.path, image_id, width, height, True)
        self.entries[entry.index - 1] = updated
        return image

    def _next_image_id(self) -> int:
        return max([0, *(int(image["id"]) for image in self.images)]) + 1

    def _next_annotation_id(self) -> int:
        return max([0, *(int(ann["id"]) for ann in self.annotations)]) + 1

    def _default_category_id(self) -> int:
        if self.categories:
            return int(self.categories[0]["id"])
        return 1
