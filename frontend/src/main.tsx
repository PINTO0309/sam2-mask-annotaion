import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import {
  ArrowBigLeft,
  ArrowBigRight,
  ChevronLeft,
  ChevronRight,
  Download,
  FolderOpen,
  PaintBucket,
  Plus,
  RotateCcw,
  ScanLine,
  Trash2,
  UserMinus,
  Save,
  Sparkles
} from "lucide-react";
import "./styles.css";

type ImageSummary = {
  index: number;
  file_name: string;
  image_id: number | null;
  width: number | null;
  height: number | null;
  in_coco: boolean;
  annotation_count: number;
};

type Annotation = {
  id: number;
  image_id: number;
  category_id: number;
  area: number;
  bbox: number[];
  is_new: boolean;
  mask_png: string;
};

type ImageDetail = {
  index: number;
  file_name: string;
  image_id: number | null;
  width: number;
  height: number;
  in_coco: boolean;
  image_url: string;
  annotations: Annotation[];
};

type DrawingAction = "paint" | "erase";
type EdgeMethod = "canny" | "ddn";
type BucketSource = "mask" | "edge";
type UndoSnapshot = { annotationId: number; maskPng: string };
type UndoStacks = Record<number, UndoSnapshot[][]>;
type CanvasPoint = { x: number; y: number };
type PanOffset = { x: number; y: number };
type PanState = { pointerId: number; clientX: number; clientY: number; offset: PanOffset };
type BucketRegion = { width: number; height: number; pixels: Uint32Array; count: number };
type EdgeBarrier = { width: number; height: number; pixels: Uint8Array; visiblePixels: Uint8Array };
type BucketRegionResult =
  | { kind: "fill"; region: BucketRegion }
  | { kind: "selected" }
  | { kind: "empty" };
type SAM2Model = {
  id: string;
  label: string;
  checkpoint_path: string;
  available: boolean;
};

const API = "";
const colorPalette = [
  [230, 75, 75],
  [39, 144, 204],
  [65, 168, 95],
  [232, 159, 54],
  [142, 99, 206],
  [219, 85, 151],
  [62, 176, 169],
  [174, 132, 54],
  [100, 117, 220],
  [112, 162, 64]
];

function dilateBarrier(source: Uint8Array, width: number, height: number) {
  const result = new Uint8Array(source.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const index = y * width + x;
      let value = 0;
      for (let dy = -1; dy <= 1 && !value; dy += 1) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          if (source[ny * width + nx]) {
            value = 1;
            break;
          }
        }
      }
      result[index] = value;
    }
  }
  return result;
}

function erodeBarrier(source: Uint8Array, width: number, height: number) {
  const result = new Uint8Array(source.length);
  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      let value = 1;
      for (let dy = -1; dy <= 1 && value; dy += 1) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) {
          value = 0;
          break;
        }
        for (let dx = -1; dx <= 1; dx += 1) {
          const nx = x + dx;
          if (nx < 0 || nx >= width || !source[ny * width + nx]) {
            value = 0;
            break;
          }
        }
      }
      result[y * width + x] = value;
    }
  }
  return result;
}

function buildEdgeBarrier(data: Uint8ClampedArray, width: number, height: number): EdgeBarrier {
  const base = new Uint8Array(width * height);
  for (let index = 0; index < base.length; index += 1) {
    base[index] = data[index * 4] >= 48 ? 1 : 0;
  }
  const closed = erodeBarrier(dilateBarrier(base, width, height), width, height);
  return { width, height, pixels: dilateBarrier(closed, width, height), visiblePixels: closed };
}

function waitForNextPaint() {
  return new Promise<void>((resolve) => {
    requestAnimationFrame(() => {
      requestAnimationFrame(() => resolve());
    });
  });
}

function wait(ms: number) {
  return new Promise<void>((resolve) => window.setTimeout(resolve, ms));
}

function App() {
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [detail, setDetail] = useState<ImageDetail | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [hoveredInstanceId, setHoveredInstanceId] = useState<number | null>(null);
  const [sam2Support, setSam2Support] = useState(false);
  const [brushSize, setBrushSize] = useState(1);
  const [zoom, setZoom] = useState(1);
  const [jumpValue, setJumpValue] = useState("1");
  const [undoStacks, setUndoStacks] = useState<UndoStacks>({});
  const [status, setStatus] = useState("Loading");
  const [drawingAction, setDrawingAction] = useState<DrawingAction | null>(null);
  const [dirtyIds, setDirtyIds] = useState<Set<number>>(new Set());
  const [sam2Models, setSam2Models] = useState<SAM2Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("tiny");
  const [blockingMessage, setBlockingMessage] = useState<string | null>(null);
  const [cursorPoint, setCursorPoint] = useState<CanvasPoint | null>(null);
  const [panOffset, setPanOffset] = useState<PanOffset>({ x: 0, y: 0 });
  const [ctrlPressed, setCtrlPressed] = useState(false);
  const [bucketEnabled, setBucketEnabled] = useState(false);
  const [bucketSource, setBucketSource] = useState<BucketSource>("edge");
  const [edgeEnabled, setEdgeEnabled] = useState(false);
  const [edgeLowThreshold, setEdgeLowThreshold] = useState(80);
  const [edgeHighThreshold, setEdgeHighThreshold] = useState(160);
  const [edgeOpacity, setEdgeOpacity] = useState(0.85);
  const [edgePng, setEdgePng] = useState<string | null>(null);
  const [edgeMethod, setEdgeMethod] = useState<EdgeMethod>("canny");
  const [ddnEdgeThickness, setDdnEdgeThickness] = useState(1);

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const workspaceRef = useRef<HTMLElement | null>(null);
  const annotationFileInputRef = useRef<HTMLInputElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const edgeCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const edgeBarrierRef = useRef<EdgeBarrier | null>(null);
  const maskCanvases = useRef<Map<number, HTMLCanvasElement>>(new Map());
  const panStateRef = useRef<PanState | null>(null);
  const bucketPreviewRef = useRef<BucketRegion | null>(null);
  const bucketPreviewStatusRef = useRef("");
  const dirtyIdsRef = useRef<Set<number>>(new Set());
  const syncDirtyMasksRef = useRef<() => Promise<void>>(async () => {});

  const selectedAnnotation = useMemo(
    () => detail?.annotations.find((annotation) => annotation.id === selectedId) ?? null,
    [detail, selectedId]
  );

  const loadImages = useCallback(async () => {
    const response = await fetch(`${API}/api/images`);
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    setImages(data.images);
    return data.images as ImageSummary[];
  }, []);

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Control") setCtrlPressed(true);
    };
    const handleKeyUp = (event: KeyboardEvent) => {
      if (event.key === "Control") setCtrlPressed(false);
    };
    const handleBlur = () => setCtrlPressed(false);
    window.addEventListener("keydown", handleKeyDown);
    window.addEventListener("keyup", handleKeyUp);
    window.addEventListener("blur", handleBlur);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      window.removeEventListener("keyup", handleKeyUp);
      window.removeEventListener("blur", handleBlur);
    };
  }, []);

  const loadSam2Models = useCallback(async () => {
    const response = await fetch(`${API}/api/sam2/models`);
    if (!response.ok) throw new Error(await response.text());
    const data = await response.json();
    setSam2Models(data.models);
    setSelectedModelId(data.current_model_id);
  }, []);

  const canvasToMaskPng = useCallback((annotationId: number) => {
    const canvas = maskCanvases.current.get(annotationId);
    if (!canvas) throw new Error(`mask canvas not found for annotation ${annotationId}`);
    return canvas.toDataURL("image/png");
  }, []);

  const syncDirtyMasks = useCallback(async () => {
    const ids = [...dirtyIdsRef.current];
    for (const annotationId of ids) {
      const response = await fetch(`${API}/api/annotations/${annotationId}/mask`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mask_png: canvasToMaskPng(annotationId) })
      });
      if (!response.ok) throw new Error(await response.text());
    }
    dirtyIdsRef.current = new Set();
    setDirtyIds(new Set());
  }, [canvasToMaskPng]);

  useEffect(() => {
    syncDirtyMasksRef.current = syncDirtyMasks;
  }, [syncDirtyMasks]);

  const draw = useCallback(() => {
    if (!detail || !canvasRef.current || !imageRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = Math.max(1, Math.round(detail.width * zoom));
    canvas.height = Math.max(1, Math.round(detail.height * zoom));
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(imageRef.current, 0, 0, canvas.width, canvas.height);

    detail.annotations.forEach((annotation, idx) => {
      const maskCanvas = maskCanvases.current.get(annotation.id);
      if (!maskCanvas) return;
      const color = colorPalette[idx % colorPalette.length];
      const isHovered = annotation.id === hoveredInstanceId;
      const maskCtx = maskCanvas.getContext("2d");
      if (!maskCtx) return;
      const source = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
      const overlay = ctx.createImageData(maskCanvas.width, maskCanvas.height);
      for (let i = 0; i < source.data.length; i += 4) {
        if (source.data[i] > 127) {
          overlay.data[i] = color[0];
          overlay.data[i + 1] = color[1];
          overlay.data[i + 2] = color[2];
          overlay.data[i + 3] = isHovered ? 210 : annotation.id === selectedId ? 150 : 95;
        }
      }
      const tinted = document.createElement("canvas");
      tinted.width = maskCanvas.width;
      tinted.height = maskCanvas.height;
      tinted.getContext("2d")?.putImageData(overlay, 0, 0);
      if (isHovered) {
        ctx.save();
        ctx.shadowColor = `rgb(${color.join(",")})`;
        ctx.shadowBlur = Math.max(8, 8 * zoom);
        ctx.drawImage(tinted, 0, 0, canvas.width, canvas.height);
        ctx.restore();
      }
      ctx.drawImage(tinted, 0, 0, canvas.width, canvas.height);
    });

    if (edgeEnabled && edgeCanvasRef.current) {
      ctx.save();
      ctx.globalAlpha = edgeOpacity;
      ctx.drawImage(edgeCanvasRef.current, 0, 0, canvas.width, canvas.height);
      ctx.restore();
    }

    if (bucketPreviewRef.current && selectedId) {
      const selectedIndex = detail.annotations.findIndex((annotation) => annotation.id === selectedId);
      const previewColor = selectedIndex >= 0 ? colorPalette[selectedIndex % colorPalette.length] : [255, 220, 74];
      const preview = bucketPreviewRef.current;
      const overlay = ctx.createImageData(preview.width, preview.height);
      for (let i = 0; i < preview.count; i += 1) {
        const offset = preview.pixels[i] * 4;
        overlay.data[offset] = previewColor[0];
        overlay.data[offset + 1] = previewColor[1];
        overlay.data[offset + 2] = previewColor[2];
        overlay.data[offset + 3] = 185;
      }
      const tinted = document.createElement("canvas");
      tinted.width = preview.width;
      tinted.height = preview.height;
      tinted.getContext("2d")?.putImageData(overlay, 0, 0);
      ctx.drawImage(tinted, 0, 0, canvas.width, canvas.height);
    }

    if (cursorPoint) {
      const scaleX = canvas.width / detail.width;
      const scaleY = canvas.height / detail.height;
      const x = cursorPoint.x * scaleX;
      const y = cursorPoint.y * scaleY;
      const radius = Math.max(1, (brushSize * scaleX) / 2);

      ctx.save();
      ctx.strokeStyle = "#000000";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(x - radius - 5, y);
      ctx.lineTo(x - 2, y);
      ctx.moveTo(x + 2, y);
      ctx.lineTo(x + radius + 5, y);
      ctx.moveTo(x, y - radius - 5);
      ctx.lineTo(x, y - 2);
      ctx.moveTo(x, y + 2);
      ctx.lineTo(x, y + radius + 5);
      ctx.stroke();

      ctx.restore();
    }
  }, [brushSize, cursorPoint, detail, edgeEnabled, edgeOpacity, hoveredInstanceId, sam2Support, selectedId, zoom]);

  const loadDetail = useCallback(
    async (index: number, imageCount = images.length) => {
      setStatus("Loading image");
      await syncDirtyMasksRef.current();
      const clamped = Math.min(Math.max(index, 1), Math.max(imageCount, 1));
      const response = await fetch(`${API}/api/images/${clamped}`);
      if (!response.ok) throw new Error(await response.text());
      const nextDetail: ImageDetail = await response.json();
      const image = new Image();
      image.src = nextDetail.image_url;
      await image.decode();
      imageRef.current = image;

      const nextMasks = new Map<number, HTMLCanvasElement>();
      await Promise.all(
        nextDetail.annotations.map(async (annotation) => {
          const maskImage = new Image();
          maskImage.src = annotation.mask_png;
          await maskImage.decode();
          const canvas = document.createElement("canvas");
          canvas.width = nextDetail.width;
          canvas.height = nextDetail.height;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.fillStyle = "black";
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(maskImage, 0, 0);
          }
          nextMasks.set(annotation.id, canvas);
        })
      );
      maskCanvases.current = nextMasks;
      edgeCanvasRef.current = null;
      edgeBarrierRef.current = null;
      bucketPreviewRef.current = null;
      setDetail(nextDetail);
      setEdgePng(null);
      setSelectedId(nextDetail.annotations[0]?.id ?? null);
      setHoveredInstanceId(null);
      setJumpValue(String(nextDetail.index));
      setUndoStacks({});
      dirtyIdsRef.current = new Set();
      setDirtyIds(new Set());
      setStatus("Ready");
    },
    [images.length]
  );

  useEffect(() => {
    loadImages().catch((error) => setStatus(`Image list error: ${error.message}`));
    loadSam2Models().catch((error) => setStatus(`SAM2 model list error: ${error.message}`));
  }, [loadImages, loadSam2Models]);

  useEffect(() => {
    if (images.length > 0 && !detail) {
      loadDetail(1).catch((error) => setStatus(`Image load error: ${error.message}`));
    }
  }, [detail, images.length, loadDetail]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    if (!edgePng) {
      edgeCanvasRef.current = null;
      edgeBarrierRef.current = null;
      draw();
      return;
    }

    let active = true;
    const image = new Image();
    image.src = edgePng;
    image.decode()
      .then(() => {
        if (!active) return;
        const canvas = document.createElement("canvas");
        canvas.width = image.width;
        canvas.height = image.height;
        const ctx = canvas.getContext("2d");
        if (!ctx) return;
        ctx.drawImage(image, 0, 0);
        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
        const { data } = imageData;
        edgeBarrierRef.current = buildEdgeBarrier(data, canvas.width, canvas.height);
        for (let i = 0; i < data.length; i += 4) {
          const edge = data[i];
          data[i] = 92;
          data[i + 1] = 226;
          data[i + 2] = 255;
          data[i + 3] = edge;
        }
        ctx.putImageData(imageData, 0, 0);
        edgeCanvasRef.current = canvas;
        draw();
      })
      .catch((error) => {
        if (!active) return;
        edgeCanvasRef.current = null;
        edgeBarrierRef.current = null;
        setStatus(`Edge error: ${error instanceof Error ? error.message : String(error)}`);
        draw();
      });

    return () => {
      active = false;
    };
  }, [edgePng]);

  useEffect(() => {
    if (!edgeEnabled || !detail) {
      setEdgePng(null);
      edgeCanvasRef.current = null;
      edgeBarrierRef.current = null;
      return;
    }

    const controller = new AbortController();
    const timeoutId = window.setTimeout(async () => {
      if (edgeMethod === "canny" && edgeLowThreshold >= edgeHighThreshold) {
        setEdgePng(null);
        edgeCanvasRef.current = null;
        edgeBarrierRef.current = null;
        setStatus("Edge error: low threshold must be less than high threshold");
        return;
      }

      if (edgeMethod === "ddn") {
        setBlockingMessage("DDN edge detection");
        await waitForNextPaint();
      }
      try {
        const params = new URLSearchParams({
          method: edgeMethod,
          low_threshold: String(edgeLowThreshold),
          high_threshold: String(edgeHighThreshold),
          ddn_thickness: String(ddnEdgeThickness),
          ddn_model: "m36"
        });
        const response = await fetch(`${API}/api/images/${detail.index}/edges?${params}`, {
          signal: controller.signal
        });
        if (!response.ok) throw new Error(await response.text());
        const data: {
          edge_png: string;
          edge_count: number;
          method: EdgeMethod;
          requested_method?: EdgeMethod;
          fallback?: boolean;
          warning?: string;
          thickness?: number;
          model?: string;
          encoder?: string;
        } = await response.json();
        setEdgePng(data.edge_png);
        if (data.fallback) {
          setStatus(`DDN fallback to Canny: ${data.warning ?? "DDN unavailable"}`);
        } else {
          const thickness = data.method === "ddn" && data.thickness ? `, ${data.thickness}px` : "";
          const model = data.method === "ddn" && data.model ? ` ${data.model.toUpperCase()}` : "";
          setStatus(`${data.method.toUpperCase()}${model} edge overlay ready: ${data.edge_count} px${thickness}`);
        }
      } catch (error) {
        if (error instanceof DOMException && error.name === "AbortError") return;
        setEdgePng(null);
        edgeCanvasRef.current = null;
        edgeBarrierRef.current = null;
        setStatus(`Edge error: ${error instanceof Error ? error.message : String(error)}`);
      } finally {
        if (edgeMethod === "ddn") setBlockingMessage(null);
      }
    }, 250);

    return () => {
      controller.abort();
      window.clearTimeout(timeoutId);
      if (edgeMethod === "ddn") setBlockingMessage(null);
    };
  }, [ddnEdgeThickness, detail, edgeEnabled, edgeHighThreshold, edgeLowThreshold, edgeMethod]);

  const moveImage = useCallback(
    (delta: number) => {
      if (!detail) return;
      loadDetail(detail.index + delta).catch((error) => setStatus(`Image load error: ${error.message}`));
    },
    [detail, loadDetail]
  );

  const jumpToValue = useCallback(() => {
    const index = Number.parseInt(jumpValue, 10);
    if (!Number.isFinite(index)) {
      setJumpValue(detail ? String(detail.index) : "1");
      return;
    }
    loadDetail(index).catch((error) => setStatus(`Image load error: ${error.message}`));
  }, [detail, jumpValue, loadDetail]);

  const markDirty = useCallback((annotationId: number) => {
    dirtyIdsRef.current = new Set(dirtyIdsRef.current).add(annotationId);
    setDirtyIds((prev) => new Set(prev).add(annotationId));
  }, []);

  const pushUndo = useCallback(
    (anchorAnnotationId: number, annotationIds: number[] = [anchorAnnotationId]) => {
      const snapshots = annotationIds
        .filter((annotationId, index, ids) => ids.indexOf(annotationId) === index)
        .map((annotationId) => ({ annotationId, maskPng: canvasToMaskPng(annotationId) }));
      setUndoStacks((prev) => ({
        ...prev,
        [anchorAnnotationId]: [...(prev[anchorAnnotationId] ?? []), snapshots]
      }));
    },
    [canvasToMaskPng]
  );

  const replaceMask = useCallback(
    async (annotationId: number, maskPng: string) => {
      if (!detail) return;
      const maskImage = new Image();
      maskImage.src = maskPng;
      await maskImage.decode();
      let canvas = maskCanvases.current.get(annotationId);
      if (!canvas) {
        canvas = document.createElement("canvas");
        canvas.width = detail.width;
        canvas.height = detail.height;
        maskCanvases.current.set(annotationId, canvas);
      }
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.fillStyle = "black";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(maskImage, 0, 0, detail.width, detail.height);
      markDirty(annotationId);
      draw();
    },
    [detail, draw, markDirty]
  );

  const canvasPoint = useCallback((event: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = event.currentTarget;
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width / zoom,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height / zoom
    };
  }, [zoom]);

  const editMaskAt = useCallback(
    (x: number, y: number, action: DrawingAction) => {
      if (!selectedId) return;
      const canvas = maskCanvases.current.get(selectedId);
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx) return;
      const radius = Math.max(0.5, brushSize / 2);
      ctx.fillStyle = action === "paint" ? "white" : "black";
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
      if (action === "paint" && detail) {
        detail.annotations.forEach((annotation) => {
          if (annotation.id === selectedId) return;
          const otherCanvas = maskCanvases.current.get(annotation.id);
          const otherCtx = otherCanvas?.getContext("2d");
          if (!otherCanvas || !otherCtx) return;
          otherCtx.fillStyle = "black";
          otherCtx.beginPath();
          otherCtx.arc(x, y, radius, 0, Math.PI * 2);
          otherCtx.fill();
          markDirty(annotation.id);
        });
      }
      markDirty(selectedId);
      draw();
    },
    [brushSize, detail, draw, markDirty, selectedId]
  );

  const collectBucketRegion = useCallback((x: number, y: number): BucketRegionResult => {
    if (!detail || !selectedId) return { kind: "empty" };
    const canvas = maskCanvases.current.get(selectedId);
    if (!canvas) return { kind: "empty" };

    const { width, height } = canvas;
    if (width <= 0 || height <= 0) return { kind: "empty" };

    const startX = Math.floor(x);
    const startY = Math.floor(y);
    if (startX < 0 || startX >= width || startY < 0 || startY >= height) return { kind: "empty" };

    const pixelCount = width * height;
    const selectedMask = new Uint8Array(pixelCount);
    const otherMasks: Uint8Array[] = [];
    const otherMaskUnion = new Uint8Array(pixelCount);
    const visited = new Uint8Array(pixelCount);
    const queue = new Uint32Array(pixelCount);
    let head = 0;
    let tail = 0;

    for (const annotation of detail.annotations) {
      const maskCanvas = maskCanvases.current.get(annotation.id);
      const maskCtx = maskCanvas?.getContext("2d");
      if (!maskCanvas || !maskCtx || maskCanvas.width !== width || maskCanvas.height !== height) continue;
      const maskData = maskCtx.getImageData(0, 0, width, height).data;
      const targetMask = annotation.id === selectedId ? selectedMask : new Uint8Array(pixelCount);
      for (let index = 0; index < pixelCount; index += 1) {
        if (maskData[index * 4] > 127) targetMask[index] = 1;
      }
      if (annotation.id !== selectedId) {
        for (let index = 0; index < pixelCount; index += 1) {
          if (targetMask[index]) otherMaskUnion[index] = 1;
        }
        otherMasks.push(targetMask);
      }
    }

    const startIndex = startY * width + startX;
    const sourceMasks = otherMasks.filter((mask) => mask[startIndex]);
    if (selectedMask[startIndex] && sourceMasks.length === 0) {
      return { kind: "selected" };
    }
    const canFill = sourceMasks.length > 0
      ? selectedMask[startIndex]
        ? (index: number) => selectedMask[index] === 1 && sourceMasks.some((mask) => mask[index] === 1)
        : (index: number) => sourceMasks.some((mask) => mask[index] === 1)
      : (index: number) => selectedMask[index] === 0 && otherMaskUnion[index] === 0;

    const enqueue = (x: number, y: number) => {
      const index = y * width + x;
      if (visited[index] || !canFill(index)) return;
      visited[index] = 1;
      queue[tail] = index;
      tail += 1;
    };

    enqueue(startX, startY);
    while (head < tail) {
      const index = queue[head];
      head += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      if (x > 0) enqueue(x - 1, y);
      if (x + 1 < width) enqueue(x + 1, y);
      if (y > 0) enqueue(x, y - 1);
      if (y + 1 < height) enqueue(x, y + 1);
    }

    return { kind: "fill", region: { width, height, pixels: queue.slice(0, tail), count: tail } };
  }, [detail, selectedId]);

  const collectEdgeBucketRegion = useCallback((x: number, y: number): BucketRegion | null => {
    if (!detail || !selectedId) return null;
    const barrier = edgeBarrierRef.current;
    if (!barrier || barrier.width !== detail.width || barrier.height !== detail.height) return null;

    const { width, height } = barrier;
    const startX = Math.floor(x);
    const startY = Math.floor(y);
    if (startX < 0 || startX >= width || startY < 0 || startY >= height) return null;

    const startIndex = startY * width + startX;
    if (barrier.pixels[startIndex]) return null;

    const visited = new Uint8Array(width * height);
    const queue = new Uint32Array(width * height);
    let head = 0;
    let tail = 0;

    const enqueue = (x: number, y: number) => {
      const index = y * width + x;
      if (visited[index] || barrier.pixels[index]) return;
      visited[index] = 1;
      queue[tail] = index;
      tail += 1;
    };

    enqueue(startX, startY);
    while (head < tail) {
      const index = queue[head];
      head += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      if (x > 0) enqueue(x - 1, y);
      if (x + 1 < width) enqueue(x + 1, y);
      if (y > 0) enqueue(x, y - 1);
      if (y + 1 < height) enqueue(x, y + 1);
    }

    if (tail === 0) return null;
    const regionMask = new Uint8Array(width * height);
    const pixels = new Uint32Array(width * height);
    let count = 0;
    for (let i = 0; i < tail; i += 1) {
      const index = queue[i];
      regionMask[index] = 1;
      pixels[count] = index;
      count += 1;
    }
    for (let i = 0; i < tail; i += 1) {
      const index = queue[i];
      const x = index % width;
      const y = Math.floor(index / width);
      for (let dy = -1; dy <= 1; dy += 1) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx += 1) {
          if (dx === 0 && dy === 0) continue;
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const neighbor = ny * width + nx;
          if (!barrier.visiblePixels[neighbor] || regionMask[neighbor]) continue;
          regionMask[neighbor] = 1;
          pixels[count] = neighbor;
          count += 1;
        }
      }
    }
    return { width, height, pixels: pixels.slice(0, count), count };
  }, [detail, selectedId]);

  const paintBucketRegion = useCallback((region: BucketRegion, statusMessage: string) => {
    if (!detail || !selectedId) return;
    const canvas = maskCanvases.current.get(selectedId);
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;

    pushUndo(selectedId, detail.annotations.map((annotation) => annotation.id));
    const imageData = ctx.getImageData(0, 0, region.width, region.height);
    const { data } = imageData;
    for (let i = 0; i < region.count; i += 1) {
      const offset = region.pixels[i] * 4;
      data[offset] = 255;
      data[offset + 1] = 255;
      data[offset + 2] = 255;
      data[offset + 3] = 255;
    }

    ctx.putImageData(imageData, 0, 0);
    detail.annotations.forEach((annotation) => {
      if (annotation.id === selectedId) return;
      const otherCanvas = maskCanvases.current.get(annotation.id);
      const otherCtx = otherCanvas?.getContext("2d");
      if (!otherCanvas || !otherCtx) return;
      const otherImageData = otherCtx.getImageData(0, 0, region.width, region.height);
      const otherData = otherImageData.data;
      for (let i = 0; i < region.count; i += 1) {
        const offset = region.pixels[i] * 4;
        otherData[offset] = 0;
        otherData[offset + 1] = 0;
        otherData[offset + 2] = 0;
        otherData[offset + 3] = 255;
      }
      otherCtx.putImageData(otherImageData, 0, 0);
      markDirty(annotation.id);
    });
    bucketPreviewRef.current = null;
    markDirty(selectedId);
    draw();
    setStatus(statusMessage);
  }, [detail, draw, markDirty, pushUndo, selectedId]);

  const fillSelectedMaskHoleAt = useCallback((x: number, y: number) => {
    if (!detail || !selectedId) return;

    const result = collectBucketRegion(x, y);
    if (result.kind === "selected") {
      setStatus("Clicked point is already in the selected mask");
      return;
    }
    if (result.kind !== "fill") return;

    paintBucketRegion(result.region, `Filled enclosed unmasked area: ${result.region.count} px`);
  }, [collectBucketRegion, detail, paintBucketRegion, selectedId]);

  const fillEdgeBucketRegionAt = useCallback((x: number, y: number) => {
    if (!detail || !selectedId) return;
    const region = collectEdgeBucketRegion(x, y);
    if (!region) {
      setStatus(edgeBarrierRef.current ? "Edge bucket: no closed edge region" : "Edge bucket: edge overlay is loading");
      return;
    }
    paintBucketRegion(region, `Filled edge region: ${region.count} px`);
  }, [collectEdgeBucketRegion, detail, edgeEnabled, paintBucketRegion, selectedId]);

  const collectMaskedBucketRegion = useCallback((x: number, y: number): BucketRegion | null => {
    if (!detail || !selectedId) return null;
    const canvas = maskCanvases.current.get(selectedId);
    if (!canvas) return null;

    const { width, height } = canvas;
    if (width <= 0 || height <= 0) return null;

    const startX = Math.floor(x);
    const startY = Math.floor(y);
    if (startX < 0 || startX >= width || startY < 0 || startY >= height) return null;

    const pixelCount = width * height;
    const maskUnion = new Uint8Array(pixelCount);
    const visited = new Uint8Array(pixelCount);
    const queue = new Uint32Array(pixelCount);
    let head = 0;
    let tail = 0;

    for (const annotation of detail.annotations) {
      const maskCanvas = maskCanvases.current.get(annotation.id);
      const maskCtx = maskCanvas?.getContext("2d");
      if (!maskCanvas || !maskCtx || maskCanvas.width !== width || maskCanvas.height !== height) continue;
      const maskData = maskCtx.getImageData(0, 0, width, height).data;
      for (let index = 0; index < pixelCount; index += 1) {
        if (maskData[index * 4] > 127) maskUnion[index] = 1;
      }
    }

    const startIndex = startY * width + startX;
    if (!maskUnion[startIndex]) return null;

    const enqueue = (x: number, y: number) => {
      const index = y * width + x;
      if (visited[index] || !maskUnion[index]) return;
      visited[index] = 1;
      queue[tail] = index;
      tail += 1;
    };

    enqueue(startX, startY);
    while (head < tail) {
      const index = queue[head];
      head += 1;
      const x = index % width;
      const y = Math.floor(index / width);
      if (x > 0) enqueue(x - 1, y);
      if (x + 1 < width) enqueue(x + 1, y);
      if (y > 0) enqueue(x, y - 1);
      if (y + 1 < height) enqueue(x, y + 1);
    }

    return { width, height, pixels: queue.slice(0, tail), count: tail };
  }, [detail, selectedId]);

  const clearBucketRegion = useCallback((region: BucketRegion, statusMessage: string) => {
    if (!detail || !selectedId) return;
    pushUndo(selectedId, detail.annotations.map((annotation) => annotation.id));
    detail.annotations.forEach((annotation) => {
      const maskCanvas = maskCanvases.current.get(annotation.id);
      const maskCtx = maskCanvas?.getContext("2d");
      if (!maskCanvas || !maskCtx) return;
      const imageData = maskCtx.getImageData(0, 0, region.width, region.height);
      const { data } = imageData;
      for (let i = 0; i < region.count; i += 1) {
        const offset = region.pixels[i] * 4;
        data[offset] = 0;
        data[offset + 1] = 0;
        data[offset + 2] = 0;
        data[offset + 3] = 255;
      }
      maskCtx.putImageData(imageData, 0, 0);
      markDirty(annotation.id);
    });

    bucketPreviewRef.current = null;
    draw();
    setStatus(statusMessage);
  }, [detail, draw, markDirty, pushUndo, selectedId]);

  const unmaskBucketRegionAt = useCallback((x: number, y: number) => {
    if (!detail || !selectedId) return;
    const region = collectMaskedBucketRegion(x, y);
    if (!region) {
      setStatus("Bucket unmask: no mask at clicked point");
      return;
    }

    clearBucketRegion(region, `Unmasked bucket region: ${region.count} px`);
  }, [clearBucketRegion, collectMaskedBucketRegion, detail, selectedId]);

  const unmaskEdgeBucketRegionAt = useCallback((x: number, y: number) => {
    if (!detail || !selectedId) return;
    const region = collectEdgeBucketRegion(x, y);
    if (!region) {
      setStatus(edgeBarrierRef.current ? "Edge bucket unmask: no closed edge region" : "Edge bucket: edge overlay is loading");
      return;
    }

    clearBucketRegion(region, `Unmasked edge region: ${region.count} px`);
  }, [clearBucketRegion, collectEdgeBucketRegion, detail, edgeEnabled, selectedId]);

  const runSam2 = useCallback(
    async (x: number, y: number) => {
      if (!detail || !selectedId) return;
      pushUndo(selectedId);
      setStatus("SAM2 predicting");
      const response = await fetch(`${API}/api/sam2/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_index: detail.index, points: [{ x, y, label: 1 }], model_id: selectedModelId })
      });
      if (!response.ok) {
        const message = await response.text();
        setStatus(`SAM2 error: ${message}`);
        return;
      }
      const data = await response.json();
      await replaceMask(selectedId, data.mask_png);
      setStatus("SAM2 mask updated");
    },
    [detail, pushUndo, replaceMask, selectedId, selectedModelId]
  );

  const selectSam2Model = useCallback(
    async (modelId: string) => {
      const targetModel = sam2Models.find((model) => model.id === modelId);
      const shouldPrepare = sam2Support || !targetModel?.available;
      const message = shouldPrepare ? `Downloading SAM2 model: ${modelId}` : `Switching SAM2 model: ${modelId}`;
      setStatus(message);
      if (shouldPrepare) {
        flushSync(() => setBlockingMessage(message));
        await waitForNextPaint();
      }
      try {
        const responsePromise = fetch(`${API}/api/sam2/models/${shouldPrepare ? "prepare" : "select"}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_id: modelId })
        });
        const [response] = await Promise.all([responsePromise, shouldPrepare ? wait(500) : Promise.resolve()]);
        if (!response.ok) {
          setStatus(`SAM2 model error: ${await response.text()}`);
          return;
        }
        const data = await response.json();
        setSam2Models(data.models);
        setSelectedModelId(data.current_model_id);
        setStatus(sam2Support ? `SAM2 support ready: ${data.current_model_id}` : `SAM2 model ready: ${data.current_model_id}`);
      } finally {
        setBlockingMessage(null);
      }
    },
    [sam2Models, sam2Support]
  );

  const toggleSam2Support = useCallback(async () => {
    if (sam2Support) {
      setSam2Support(false);
      setStatus("SAM2 support off");
      return;
    }
    const message = `Downloading SAM2 model: ${selectedModelId}`;
    setStatus(message);
    flushSync(() => setBlockingMessage(message));
    await waitForNextPaint();
    try {
      const responsePromise = fetch(`${API}/api/sam2/models/prepare`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: selectedModelId })
      });
      const [response] = await Promise.all([responsePromise, wait(500)]);
      if (!response.ok) {
        setStatus(`SAM2 prepare error: ${await response.text()}`);
        return;
      }
      const data = await response.json();
      setSam2Models(data.models);
      setSelectedModelId(data.current_model_id);
      setSam2Support(true);
      setStatus(`SAM2 support on: ${data.current_model_id}`);
    } finally {
      setBlockingMessage(null);
    }
  }, [sam2Support, selectedModelId]);

  const startPan = useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (event.button !== 0 || !event.ctrlKey) return false;
    event.preventDefault();
    event.stopPropagation();
    panStateRef.current = {
      pointerId: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      offset: panOffset
    };
    setCursorPoint(null);
    bucketPreviewRef.current = null;
    setDrawingAction(null);
    try {
      event.currentTarget.setPointerCapture(event.pointerId);
    } catch {
      // Window-level listeners below keep pan working even if capture is unavailable.
    }
    setStatus("Pan");
    return true;
  }, [panOffset]);

  useEffect(() => {
    const handleWindowPointerMove = (event: PointerEvent) => {
      if (!panStateRef.current) return;
      event.preventDefault();
      const deltaX = event.clientX - panStateRef.current.clientX;
      const deltaY = event.clientY - panStateRef.current.clientY;
      setPanOffset({
        x: panStateRef.current.offset.x + deltaX,
        y: panStateRef.current.offset.y + deltaY
      });
    };
    const handleWindowPointerUp = () => {
      panStateRef.current = null;
    };
    window.addEventListener("pointermove", handleWindowPointerMove, { passive: false });
    window.addEventListener("pointerup", handleWindowPointerUp);
    window.addEventListener("pointercancel", handleWindowPointerUp);
    return () => {
      window.removeEventListener("pointermove", handleWindowPointerMove);
      window.removeEventListener("pointerup", handleWindowPointerUp);
      window.removeEventListener("pointercancel", handleWindowPointerUp);
    };
  }, []);

  const panCanvas = useCallback((event: React.PointerEvent<HTMLElement>) => {
    if (!panStateRef.current) return false;
    const deltaX = event.clientX - panStateRef.current.clientX;
    const deltaY = event.clientY - panStateRef.current.clientY;
    setPanOffset({
      x: panStateRef.current.offset.x + deltaX,
      y: panStateRef.current.offset.y + deltaY
    });
    return true;
  }, []);

  const handleWorkspacePointerDown = useCallback(
    (event: React.PointerEvent<HTMLElement>) => {
      if (startPan(event)) {
        event.stopPropagation();
      }
    },
    [startPan]
  );

  const handleWorkspacePointerMove = useCallback(
    (event: React.PointerEvent<HTMLElement>) => {
      if (panCanvas(event)) {
        event.preventDefault();
        event.stopPropagation();
      }
    },
    [panCanvas]
  );

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (!detail || !selectedId) return;
      if (startPan(event)) return;
      event.preventDefault();
      const point = canvasPoint(event);
      const activeBucketSource = edgeEnabled ? bucketSource : "mask";
      if (event.button === 2) {
        if (bucketEnabled) {
          if (activeBucketSource === "edge") {
            unmaskEdgeBucketRegionAt(point.x, point.y);
          } else {
            unmaskBucketRegionAt(point.x, point.y);
          }
          return;
        }
        pushUndo(selectedId);
        setDrawingAction("erase");
        event.currentTarget.setPointerCapture(event.pointerId);
        editMaskAt(point.x, point.y, "erase");
        return;
      }
      if (event.button !== 0) return;
      if (bucketEnabled) {
        if (activeBucketSource === "edge") {
          fillEdgeBucketRegionAt(point.x, point.y);
        } else {
          fillSelectedMaskHoleAt(point.x, point.y);
        }
        return;
      }
      if (sam2Support) {
        runSam2(point.x, point.y).catch((error) => setStatus(`SAM2 error: ${error.message}`));
        return;
      }
      pushUndo(selectedId, detail.annotations.map((annotation) => annotation.id));
      setDrawingAction("paint");
      event.currentTarget.setPointerCapture(event.pointerId);
      editMaskAt(point.x, point.y, "paint");
    },
    [
      bucketEnabled,
      bucketSource,
      canvasPoint,
      detail,
      edgeEnabled,
      editMaskAt,
      fillEdgeBucketRegionAt,
      fillSelectedMaskHoleAt,
      pushUndo,
      runSam2,
      sam2Support,
      selectedId,
      startPan,
      unmaskBucketRegionAt,
      unmaskEdgeBucketRegionAt
    ]
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (panCanvas(event)) return;
      const point = canvasPoint(event);
      setCursorPoint(point);
      if (bucketEnabled && !drawingAction) {
        const activeBucketSource = edgeEnabled ? bucketSource : "mask";
        let nextStatus: string;
        if (activeBucketSource === "edge") {
          const region = collectEdgeBucketRegion(point.x, point.y);
          bucketPreviewRef.current = region;
          nextStatus = region
            ? `Edge bucket preview: ${region.count} px`
            : edgeBarrierRef.current
              ? "Edge bucket preview: no closed edge region"
              : "Edge bucket preview: edge loading";
        } else {
          const result = collectBucketRegion(point.x, point.y);
          bucketPreviewRef.current = result.kind === "fill" ? result.region : null;
          nextStatus =
            result.kind === "fill"
              ? `Bucket preview: ${result.region.count} px`
              : result.kind === "selected"
                ? "Bucket preview: point is already in selected mask"
                : "Bucket preview: no target";
        }
        if (bucketPreviewStatusRef.current !== nextStatus) {
          bucketPreviewStatusRef.current = nextStatus;
          setStatus(nextStatus);
        }
        draw();
        return;
      }
      bucketPreviewRef.current = null;
      bucketPreviewStatusRef.current = "";
      if (!drawingAction) {
        draw();
        return;
      }
      editMaskAt(point.x, point.y, drawingAction);
    },
    [
      bucketEnabled,
      bucketSource,
      canvasPoint,
      collectBucketRegion,
      collectEdgeBucketRegion,
      draw,
      drawingAction,
      edgeEnabled,
      editMaskAt,
      panCanvas
    ]
  );

  const endDrawing = useCallback(() => {
    panStateRef.current = null;
    setDrawingAction(null);
  }, []);

  useEffect(() => {
    const workspace = workspaceRef.current;
    if (!workspace) return;
    const handleWheel = (event: WheelEvent) => {
      if (!event.ctrlKey) return;
      event.preventDefault();
      const factor = event.deltaY < 0 ? 1.12 : 0.88;
      setZoom((value) => Math.min(12, Math.max(0.15, value * factor)));
    };
    workspace.addEventListener("wheel", handleWheel, { passive: false });
    return () => {
      workspace.removeEventListener("wheel", handleWheel);
    };
  }, []);

  const undo = useCallback(async () => {
    if (!selectedId) return;
    const stack = undoStacks[selectedId] ?? [];
    const last = stack[stack.length - 1];
    if (!last) return;
    setUndoStacks((prev) => ({
      ...prev,
      [selectedId]: (prev[selectedId] ?? []).slice(0, -1)
    }));
    for (const snapshot of last) {
      await replaceMask(snapshot.annotationId, snapshot.maskPng);
    }
    setStatus("Undo applied");
  }, [replaceMask, selectedId, undoStacks]);

  const resetAllMaskEdits = useCallback(async () => {
    const ok = window.confirm(
      "Reset all mask edits and restore the input COCO JSON state? Unsaved edits and new instances will be discarded. This action cannot be undone."
    );
    if (!ok) return;
    setStatus("Resetting mask edits");
    const currentIndex = detail?.index ?? 1;
    const response = await fetch(`${API}/api/reset-edits`, { method: "POST" });
    if (!response.ok) {
      setStatus(`Reset error: ${await response.text()}`);
      return;
    }
    maskCanvases.current = new Map();
    edgeCanvasRef.current = null;
    edgeBarrierRef.current = null;
    dirtyIdsRef.current = new Set();
    setUndoStacks({});
    setDirtyIds(new Set());
    setEdgePng(null);
    setHoveredInstanceId(null);
    await loadImages();
    await loadDetail(currentIndex);
    setStatus("Mask edits reset");
  }, [detail?.index, loadDetail, loadImages]);

  const addInstance = useCallback(async () => {
    if (!detail) return;
    const response = await fetch(`${API}/api/images/${detail.index}/instances`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({})
    });
    if (!response.ok) {
      setStatus(`Create error: ${await response.text()}`);
      return;
    }
    const annotation: Annotation = await response.json();
    const canvas = document.createElement("canvas");
    canvas.width = detail.width;
    canvas.height = detail.height;
    canvas.getContext("2d")?.fillRect(0, 0, canvas.width, canvas.height);
    maskCanvases.current.set(annotation.id, canvas);
    setDetail({ ...detail, image_id: annotation.image_id, annotations: [...detail.annotations, annotation] });
    setSelectedId(annotation.id);
    markDirty(annotation.id);
    setStatus("New instance created");
  }, [detail, markDirty]);

  const deleteSelectedInstance = useCallback(async () => {
    if (!detail || !selectedId) return;
    const ok = window.confirm(`Delete selected instance #${selectedId}? This change will be written to the corrected JSON when you save.`);
    if (!ok) return;
    const response = await fetch(`${API}/api/annotations/${selectedId}`, { method: "DELETE" });
    if (!response.ok) {
      setStatus(`Delete error: ${await response.text()}`);
      return;
    }
    const nextAnnotations = detail.annotations.filter((annotation) => annotation.id !== selectedId);
    maskCanvases.current.delete(selectedId);
    dirtyIdsRef.current.delete(selectedId);
    setDirtyIds((prev) => {
      const next = new Set(prev);
      next.delete(selectedId);
      return next;
    });
    setUndoStacks((prev) => {
      const next = { ...prev };
      delete next[selectedId];
      return next;
    });
    setDetail({ ...detail, annotations: nextAnnotations });
    setHoveredInstanceId(null);
    setSelectedId(nextAnnotations[0]?.id ?? null);
    setStatus(`Instance #${selectedId} deleted`);
  }, [detail, selectedId]);

  const save = useCallback(
    async (download: boolean) => {
      setStatus("Saving");
      try {
        await syncDirtyMasks();
        const response = await fetch(`${API}/api/save`, { method: "POST" });
        if (!response.ok) throw new Error(await response.text());
        const data = await response.json();
        setStatus(`Saved: ${data.path}`);
        if (download) {
          window.location.href = `${API}/api/download`;
        }
      } catch (error) {
        setStatus(`Save error: ${error instanceof Error ? error.message : String(error)}`);
      }
    },
    [syncDirtyMasks]
  );

  const openAnnotationFile = useCallback(
    async (file: File) => {
      if (!file.name.toLowerCase().endsWith(".json")) {
        setStatus("Open error: annotation file must have a .json extension");
        return;
      }
      const ok = window.confirm(
        "Open the selected annotation file and discard the current workspace state? Unsaved edits will be lost."
      );
      if (!ok) return;
      setStatus(`Opening annotation: ${file.name}`);
      try {
        const text = await file.text();
        let data: unknown;
        try {
          data = JSON.parse(text);
        } catch (error) {
          throw new Error(`JSON parse error: ${error instanceof Error ? error.message : String(error)}`);
        }
        if (!data || typeof data !== "object" || Array.isArray(data)) {
          throw new Error("annotation file must be a COCO JSON object");
        }
        const response = await fetch(`${API}/api/annotations/open`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ file_name: file.name, data })
        });
        if (!response.ok) {
          const text = await response.text();
          let message = text;
          try {
            const parsed = JSON.parse(text);
            message = parsed.detail ?? text;
          } catch {
            // Keep the raw response text when the server did not return JSON.
          }
          throw new Error(message);
        }
        maskCanvases.current = new Map();
        edgeCanvasRef.current = null;
        edgeBarrierRef.current = null;
        dirtyIdsRef.current = new Set();
        setUndoStacks({});
        setDirtyIds(new Set());
        setEdgePng(null);
        setSelectedId(null);
        setCursorPoint(null);
        const nextImages = await loadImages();
        await loadDetail(1, nextImages.length);
        setStatus(`Opened: ${file.name}`);
      } catch (error) {
        setStatus(`Open error: ${error instanceof Error ? error.message : String(error)}`);
      }
    },
    [loadDetail, loadImages]
  );

  const selectedColor = detail?.annotations.findIndex((annotation) => annotation.id === selectedId) ?? -1;
  const selectedUndoCount = selectedId ? undoStacks[selectedId]?.length ?? 0 : 0;

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="brand">
          <Sparkles size={20} />
          <h1>SAM2 Mask Annotation</h1>
          <button title="Reset all mask edits" onClick={() => resetAllMaskEdits().catch((error) => setStatus(`Reset error: ${error.message}`))}>
            <Trash2 size={18} />
          </button>
        </div>
        <div className="nav-row">
          <button title="-10" onClick={() => moveImage(-10)}><ArrowBigLeft size={18} /></button>
          <button title="-1" onClick={() => moveImage(-1)}><ChevronLeft size={18} /></button>
          <input
            aria-label="image index"
            value={jumpValue}
            onChange={(event) => setJumpValue(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") jumpToValue();
            }}
            onBlur={jumpToValue}
          />
          <button title="+1" onClick={() => moveImage(1)}><ChevronRight size={18} /></button>
          <button title="+10" onClick={() => moveImage(10)}><ArrowBigRight size={18} /></button>
        </div>
        <div className="meta">
          <span>{detail?.file_name ?? ""}</span>
          <span>{detail ? `${detail.index} / ${images.length}` : ""}</span>
        </div>
        <section className="tools">
          <button
            className={sam2Support ? "active" : ""}
            title="SAM2 support toggle"
            onClick={() => toggleSam2Support().catch((error) => setStatus(`SAM2 prepare error: ${error.message}`))}
          >
            <Sparkles size={18} />
          </button>
          <button
            className={edgeEnabled ? "active" : ""}
            title="Image edge overlay"
            onClick={() => {
              const nextEnabled = !edgeEnabled;
              setEdgeEnabled(nextEnabled);
              if (nextEnabled) {
                setStatus("Edge overlay loading");
              } else {
                edgeCanvasRef.current = null;
                edgeBarrierRef.current = null;
                setEdgePng(null);
                setBucketSource("mask");
                setStatus("Edge overlay off");
              }
            }}
          >
            <ScanLine size={18} />
          </button>
          <button
            className={bucketEnabled ? "active" : ""}
            title="Bucket fill regions"
            onClick={() => {
              if (bucketEnabled) {
                bucketPreviewRef.current = null;
                bucketPreviewStatusRef.current = "";
                draw();
              }
              setBucketEnabled((value) => !value);
              if (!bucketEnabled && !edgeEnabled && bucketSource === "edge") {
                setBucketSource("mask");
              }
              setStatus(
                bucketEnabled
                  ? "Bucket off"
                  : `${edgeEnabled && bucketSource === "edge" ? "Edge regions" : "Mask holes"} bucket on`
              );
            }}
            disabled={!selectedId}
          >
            <PaintBucket size={18} />
          </button>
          <button title="Delete selected instance" onClick={() => deleteSelectedInstance().catch((error) => setStatus(`Delete error: ${error.message}`))} disabled={!selectedId}>
            <UserMinus size={18} />
          </button>
          <button title="Add instance" onClick={addInstance}>
            <Plus size={18} />
          </button>
          <button title="Undo" onClick={undo} disabled={selectedUndoCount === 0}>
            <RotateCcw size={18} />
          </button>
        </section>
        <h2 className="control-heading">SAM2.1 model</h2>
        <section className="model-options" aria-label="SAM2 model">
          {sam2Models.map((model) => (
            <button
              key={model.id}
              className={model.id === selectedModelId ? "active" : ""}
              title={`${model.label}${model.available ? "" : " (download on first use)"}`}
              onClick={() => selectSam2Model(model.id)}
              disabled={!sam2Support}
            >
              {model.label}
            </button>
          ))}
        </section>
        <h2 className="control-heading">Edge Detection Method</h2>
        <section className="bucket-options" aria-label="bucket source">
          <button
            className={bucketSource === "edge" ? "active" : ""}
            title="Fill regions enclosed by the current edge overlay"
            onClick={() => {
              setBucketSource("edge");
              setBucketEnabled(true);
              setStatus("Edge regions bucket on");
              bucketPreviewRef.current = null;
              bucketPreviewStatusRef.current = "";
              draw();
            }}
            disabled={!edgeEnabled || !selectedId}
          >
            Edge
          </button>
          <button
            className={bucketSource === "mask" ? "active" : ""}
            title="Fill mask-enclosed holes"
            onClick={() => {
              setBucketSource("mask");
              setBucketEnabled(true);
              bucketPreviewRef.current = null;
              bucketPreviewStatusRef.current = "";
              draw();
              setStatus("Mask holes bucket on");
            }}
            disabled={!edgeEnabled || !selectedId}
          >
            Mask
          </button>
        </section>
        <section className="edge-options" aria-label="edge method">
          <button
            className={edgeMethod === "canny" ? "active" : ""}
            title="Use OpenCV Canny edge detection"
            onClick={() => setEdgeMethod("canny")}
            disabled={!edgeEnabled}
          >
            Canny
          </button>
          <button
            className={edgeMethod === "ddn" ? "active" : ""}
            title="Use DDN-M36 AI edge detection"
            onClick={() => setEdgeMethod("ddn")}
            disabled={!edgeEnabled}
          >
            DDN
          </button>
        </section>
        <label className="slider">
          <span>Brush</span>
          <input min={1} max={80} type="range" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
          <strong>{brushSize}px</strong>
        </label>
        <label className="slider">
          <span>Edge low</span>
          <input
            disabled={!edgeEnabled || edgeMethod === "ddn"}
            min={0}
            max={Math.max(0, edgeHighThreshold - 1)}
            type="range"
            value={edgeLowThreshold}
            onChange={(event) => setEdgeLowThreshold(Math.min(Number(event.target.value), edgeHighThreshold - 1))}
          />
          <strong>{edgeLowThreshold}</strong>
        </label>
        <label className="slider">
          <span>Edge high</span>
          <input
            disabled={!edgeEnabled || edgeMethod === "ddn"}
            min={Math.min(255, edgeLowThreshold + 1)}
            max={255}
            type="range"
            value={edgeHighThreshold}
            onChange={(event) => setEdgeHighThreshold(Math.max(Number(event.target.value), edgeLowThreshold + 1))}
          />
          <strong>{edgeHighThreshold}</strong>
        </label>
        <label className="slider">
          <span>Edge</span>
          <input
            disabled={!edgeEnabled}
            min={0.1}
            max={1}
            step={0.05}
            type="range"
            value={edgeOpacity}
            onChange={(event) => setEdgeOpacity(Number(event.target.value))}
          />
          <strong>{Math.round(edgeOpacity * 100)}%</strong>
        </label>
        <label className="slider">
          <span>DDN px</span>
          <input
            disabled={!edgeEnabled || edgeMethod !== "ddn"}
            min={1}
            max={2}
            step={1}
            type="range"
            value={ddnEdgeThickness}
            onChange={(event) => setDdnEdgeThickness(Number(event.target.value))}
          />
          <strong>{ddnEdgeThickness}px</strong>
        </label>
        <section className="legend" aria-label="mouse controls">
          <h2>Mouse</h2>
          <div><kbd>Left drag</kbd><span>Paint mask</span></div>
          <div><kbd>Right drag</kbd><span>Erase mask</span></div>
          <div><kbd>Bucket + left</kbd><span>Fill region</span></div>
          <div><kbd>Bucket + right</kbd><span>Unmask region</span></div>
          <div><kbd>Ctrl + drag</kbd><span>Pan image</span></div>
          <div><kbd>Ctrl + wheel</kbd><span>Zoom image</span></div>
          <div><kbd>SAM2 ON + left</kbd><span>SAM2 assist</span></div>
        </section>
        <section className="instances">
          {detail?.annotations.map((annotation, idx) => {
            const color = colorPalette[idx % colorPalette.length];
            return (
              <button
                key={annotation.id}
                className={annotation.id === selectedId ? "instance active" : "instance"}
                onClick={() => setSelectedId(annotation.id)}
                onFocus={() => setHoveredInstanceId(annotation.id)}
                onBlur={() => setHoveredInstanceId(null)}
                onMouseEnter={() => setHoveredInstanceId(annotation.id)}
                onMouseLeave={() => setHoveredInstanceId(null)}
              >
                <span className="swatch" style={{ backgroundColor: `rgb(${color.join(",")})` }} />
                <span>#{annotation.id}</span>
                <small>{Math.round(annotation.area)} px</small>
              </button>
            );
          })}
        </section>
        <div className="actions">
          <input
            ref={annotationFileInputRef}
            type="file"
            accept=".json,application/json"
            className="hidden-file-input"
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = "";
              if (file) {
                openAnnotationFile(file).catch((error) => setStatus(`Open error: ${error.message}`));
              }
            }}
          />
          <button title="Open annotation JSON" onClick={() => annotationFileInputRef.current?.click()}><FolderOpen size={18} /></button>
          <button title="Save" onClick={() => save(false)}><Save size={18} /></button>
          <button title="Save and download" onClick={() => save(true)}><Download size={18} /></button>
        </div>
        <p className="status">{status}</p>
      </aside>
      <section
        ref={workspaceRef}
        className={ctrlPressed || panStateRef.current ? "workspace panning" : "workspace"}
        onPointerDownCapture={handleWorkspacePointerDown}
        onPointerMoveCapture={handleWorkspacePointerMove}
        onPointerUp={endDrawing}
        onPointerCancel={endDrawing}
      >
        <div className="canvas-shell">
          <canvas
            ref={canvasRef}
            className="image-canvas"
            style={{ transform: `translate(${panOffset.x}px, ${panOffset.y}px)` }}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={endDrawing}
            onPointerCancel={endDrawing}
            onPointerLeave={() => {
              setCursorPoint(null);
              bucketPreviewRef.current = null;
              bucketPreviewStatusRef.current = "";
              draw();
              if (!panStateRef.current) {
                endDrawing();
              }
            }}
            onContextMenu={(event) => event.preventDefault()}
          />
        </div>
        <div className="footer-bar">
          <span>Zoom {Math.round(zoom * 100)}%</span>
          <span>{selectedAnnotation ? `Selected #${selectedAnnotation.id}` : "No instance"}</span>
          <span>{selectedColor >= 0 ? `Color ${selectedColor + 1}` : ""}</span>
        </div>
      </section>
      {blockingMessage && (
        <div className="blocking-overlay" role="alert" aria-live="assertive">
          <div className="blocking-panel">
            <Sparkles size={22} />
            <strong>{blockingMessage}</strong>
            <span>Please wait.</span>
          </div>
        </div>
      )}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
