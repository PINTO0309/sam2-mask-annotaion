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
        self.output_path = output_path
        with self.coco_path.open("r", encoding="utf-8") as fh:
            self.data: dict[str, Any] = json.load(fh)

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
        if annotation_id not in self._original_annotation_ids:
            ann["bbox"] = mask_bbox(mask)
        return self.annotation_response(ann)

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
