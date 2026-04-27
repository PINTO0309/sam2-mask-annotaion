import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  ArrowBigLeft,
  ArrowBigRight,
  ChevronLeft,
  ChevronRight,
  CircleMinus,
  CirclePlus,
  Download,
  Eraser,
  Plus,
  RotateCcw,
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

type Tool = "sam_positive" | "sam_negative" | "eraser";
type UndoItem = { annotationId: number; maskPng: string };
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

function App() {
  const [images, setImages] = useState<ImageSummary[]>([]);
  const [detail, setDetail] = useState<ImageDetail | null>(null);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [tool, setTool] = useState<Tool>("sam_positive");
  const [brushSize, setBrushSize] = useState(8);
  const [zoom, setZoom] = useState(1);
  const [jumpValue, setJumpValue] = useState("1");
  const [undoStack, setUndoStack] = useState<UndoItem[]>([]);
  const [status, setStatus] = useState("Loading");
  const [isDrawing, setIsDrawing] = useState(false);
  const [dirtyIds, setDirtyIds] = useState<Set<number>>(new Set());
  const [promptPoints, setPromptPoints] = useState<{ x: number; y: number; label: number }[]>([]);
  const [sam2Models, setSam2Models] = useState<SAM2Model[]>([]);
  const [selectedModelId, setSelectedModelId] = useState("tiny");

  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const maskCanvases = useRef<Map<number, HTMLCanvasElement>>(new Map());
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
      const maskCtx = maskCanvas.getContext("2d");
      if (!maskCtx) return;
      const source = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
      const overlay = ctx.createImageData(maskCanvas.width, maskCanvas.height);
      for (let i = 0; i < source.data.length; i += 4) {
        if (source.data[i] > 127) {
          overlay.data[i] = color[0];
          overlay.data[i + 1] = color[1];
          overlay.data[i + 2] = color[2];
          overlay.data[i + 3] = annotation.id === selectedId ? 150 : 95;
        }
      }
      const tinted = document.createElement("canvas");
      tinted.width = maskCanvas.width;
      tinted.height = maskCanvas.height;
      tinted.getContext("2d")?.putImageData(overlay, 0, 0);
      ctx.drawImage(tinted, 0, 0, canvas.width, canvas.height);
    });
  }, [detail, selectedId, zoom]);

  const loadDetail = useCallback(
    async (index: number) => {
      setStatus("Loading image");
      await syncDirtyMasksRef.current();
      const clamped = Math.min(Math.max(index, 1), Math.max(images.length, 1));
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
      setDetail(nextDetail);
      setSelectedId(nextDetail.annotations[0]?.id ?? null);
      setPromptPoints([]);
      setJumpValue(String(nextDetail.index));
      setZoom(1);
      setUndoStack([]);
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
    (annotationId: number) => {
      setUndoStack((prev) => [...prev, { annotationId, maskPng: canvasToMaskPng(annotationId) }]);
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

  const eraseAt = useCallback(
    (x: number, y: number) => {
      if (!selectedId) return;
      const canvas = maskCanvases.current.get(selectedId);
      const ctx = canvas?.getContext("2d");
      if (!canvas || !ctx) return;
      ctx.fillStyle = "black";
      ctx.beginPath();
      ctx.arc(x, y, Math.max(0.5, brushSize / 2), 0, Math.PI * 2);
      ctx.fill();
      markDirty(selectedId);
      draw();
    },
    [brushSize, draw, markDirty, selectedId]
  );

  const runSam2 = useCallback(
    async (x: number, y: number, label: number) => {
      if (!detail || !selectedId) return;
      pushUndo(selectedId);
      const points = [...promptPoints, { x, y, label }];
      setPromptPoints(points);
      setStatus("SAM2 predicting");
      const response = await fetch(`${API}/api/sam2/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_index: detail.index, points, model_id: selectedModelId })
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
    [detail, promptPoints, pushUndo, replaceMask, selectedId, selectedModelId]
  );

  const selectSam2Model = useCallback(
    async (modelId: string) => {
      setStatus(`Switching SAM2 model: ${modelId}`);
      const response = await fetch(`${API}/api/sam2/models/select`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_id: modelId })
      });
      if (!response.ok) {
        setStatus(`SAM2 model error: ${await response.text()}`);
        return;
      }
      const data = await response.json();
      setSam2Models(data.models);
      setSelectedModelId(data.current_model_id);
      setPromptPoints([]);
      setStatus(`SAM2 model: ${data.current_model_id}`);
    },
    []
  );

  const handlePointerDown = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (!detail || !selectedId) return;
      const point = canvasPoint(event);
      if (tool === "eraser") {
        pushUndo(selectedId);
        setIsDrawing(true);
        event.currentTarget.setPointerCapture(event.pointerId);
        eraseAt(point.x, point.y);
        return;
      }
      runSam2(point.x, point.y, tool === "sam_positive" ? 1 : 0).catch((error) =>
        setStatus(`SAM2 error: ${error.message}`)
      );
    },
    [canvasPoint, detail, eraseAt, pushUndo, runSam2, selectedId, tool]
  );

  const handlePointerMove = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>) => {
      if (!isDrawing || tool !== "eraser") return;
      const point = canvasPoint(event);
      eraseAt(point.x, point.y);
    },
    [canvasPoint, eraseAt, isDrawing, tool]
  );

  const endDrawing = useCallback(() => {
    setIsDrawing(false);
  }, []);

  const undo = useCallback(async () => {
    const last = undoStack[undoStack.length - 1];
    if (!last) return;
    setUndoStack((prev) => prev.slice(0, -1));
    await replaceMask(last.annotationId, last.maskPng);
    setStatus("Undo applied");
  }, [replaceMask, undoStack]);

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
    setPromptPoints([]);
    setStatus("New instance created");
  }, [detail, markDirty]);

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

  const selectedColor = detail?.annotations.findIndex((annotation) => annotation.id === selectedId) ?? -1;

  return (
    <main className="app">
      <aside className="sidebar">
        <div className="brand">
          <Sparkles size={20} />
          <h1>SAM2 Mask Annotation</h1>
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
          <button className={tool === "sam_positive" ? "active" : ""} title="SAM2 positive point" onClick={() => setTool("sam_positive")}>
            <CirclePlus size={18} />
          </button>
          <button className={tool === "sam_negative" ? "active" : ""} title="SAM2 negative point" onClick={() => setTool("sam_negative")}>
            <CircleMinus size={18} />
          </button>
          <button className={tool === "eraser" ? "active" : ""} title="Eraser" onClick={() => setTool("eraser")}>
            <Eraser size={18} />
          </button>
          <button title="Undo" onClick={undo} disabled={undoStack.length === 0}>
            <RotateCcw size={18} />
          </button>
          <button title="Add instance" onClick={addInstance}>
            <Plus size={18} />
          </button>
        </section>
        <section className="model-options" aria-label="SAM2 model">
          {sam2Models.map((model) => (
            <button
              key={model.id}
              className={model.id === selectedModelId ? "active" : ""}
              title={`${model.label}${model.available ? "" : " (download on first use)"}`}
              onClick={() => selectSam2Model(model.id)}
            >
              {model.label}
            </button>
          ))}
        </section>
        <label className="slider">
          <span>Brush</span>
          <input min={1} max={80} type="range" value={brushSize} onChange={(event) => setBrushSize(Number(event.target.value))} />
          <strong>{brushSize}px</strong>
        </label>
        <section className="instances">
          {detail?.annotations.map((annotation, idx) => {
            const color = colorPalette[idx % colorPalette.length];
            return (
              <button
                key={annotation.id}
                className={annotation.id === selectedId ? "instance active" : "instance"}
                onClick={() => {
                  setSelectedId(annotation.id);
                  setPromptPoints([]);
                }}
              >
                <span className="swatch" style={{ backgroundColor: `rgb(${color.join(",")})` }} />
                <span>#{annotation.id}</span>
                <small>{Math.round(annotation.area)} px</small>
              </button>
            );
          })}
        </section>
        <div className="actions">
          <button title="Save" onClick={() => save(false)}><Save size={18} /></button>
          <button title="Save and download" onClick={() => save(true)}><Download size={18} /></button>
        </div>
        <p className="status">{status}</p>
      </aside>
      <section className="workspace">
        <div className="canvas-shell">
          <canvas
            ref={canvasRef}
            className="image-canvas"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={endDrawing}
            onPointerCancel={endDrawing}
            onWheel={(event) => {
              event.preventDefault();
              const factor = event.deltaY < 0 ? 1.12 : 0.88;
              setZoom((value) => Math.min(12, Math.max(0.15, value * factor)));
            }}
          />
        </div>
        <div className="footer-bar">
          <span>Zoom {Math.round(zoom * 100)}%</span>
          <span>{selectedAnnotation ? `Selected #${selectedAnnotation.id}` : "No instance"}</span>
          <span>{selectedColor >= 0 ? `Color ${selectedColor + 1}` : ""}</span>
        </div>
      </section>
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<App />);
