#!/usr/bin/env python3
"""video-square-mcp: convierte clips a cuadrado 1080x1080 enfocando el sujeto,
corta por timecodes, une clips, mezcla música de fondo y exporta para redes.

Pipeline: cut -> reframe -> add_music/export. Envuelve FFmpeg + OpenCV.
"""

import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from fastmcp import FastMCP

mcp = FastMCP("video-square")

TARGET = 1080  # lado del cuadrado de salida
TAU = 0.8  # constante de tiempo del suavizado (s)
SAMPLE_EVERY = 0.5  # muestreo de caras (s)
FPS_CAP = 60.0

# ---------- utilidades ----------


class VideoError(Exception):
    pass


def _check_file(path: str, kind: str = "archivo") -> Path:
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        raise VideoError(f"{kind} no existe: {p}")
    return p


def _run(cmd: list[str], capture: bool = True, **kw) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=capture, text=True, **kw)
    if proc.returncode != 0:
        err = (proc.stderr or "")[-3000:] if capture else ""
        raise VideoError(
            f"comando falló ({proc.returncode}): {' '.join(cmd[:6])}...\n{err}"
        )
    return proc


def _ffprobe(path: Path) -> dict:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ]
    )
    return json.loads(proc.stdout)


def _video_info(path: Path) -> dict:
    data = _ffprobe(path)
    vstream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"), None
    )
    if vstream is None:
        raise VideoError(f"sin stream de video: {path}")
    astream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "audio"), None
    )
    fps_raw = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "30/1"
    try:
        num, den = fps_raw.split("/")
        fps = float(num) / float(den) if float(den) else 30.0
    except Exception:
        fps = 30.0
    dur = float(data.get("format", {}).get("duration") or vstream.get("duration") or 0)
    return {
        "path": str(path),
        "duration": round(dur, 3),
        "width": int(vstream["width"]),
        "height": int(vstream["height"]),
        "fps": round(fps, 3),
        "video_codec": vstream.get("codec_name"),
        "has_audio": astream is not None,
        "audio_codec": astream.get("codec_name") if astream else None,
        "bitrate": int(data.get("format", {}).get("bit_rate") or 0) or None,
    }


def _parse_time(value) -> float:
    """Acepta segundos (float/int/str numérico) o 'MM:SS[.ms]' / 'HH:MM:SS[.ms]'."""
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if re.fullmatch(r"\d+(\.\d+)?", s):
        return float(s)
    parts = s.split(":")
    try:
        parts = [float(p) for p in parts]
    except ValueError:
        raise VideoError(f"timecode inválido: {value!r}")
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    raise VideoError(f"timecode inválido: {value!r}")


def _any_audio(path: Path) -> bool:
    return _video_info(path)["has_audio"]


# ---------- detección y trayectoria ----------


_YUNET = Path(__file__).parent / "models" / "face_detection_yunet_2023mar.onnx"


def _make_detector(w: int, h: int):
    if not _YUNET.is_file():
        return None
    # redimensionamos a máx 960px de ancho para acelerar; coords se reescala en _detect_face_frame
    scale = min(1.0, 960.0 / max(w, 1))
    iw, ih = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    det = cv2.FaceDetectorYN.create(
        str(_YUNET), "", (iw, ih), score_threshold=0.6, nms_threshold=0.3, top_k=5000
    )
    return (det, iw, ih)


def _detect_face_frame(frame, detector):
    if detector is None:
        return None
    det, iw, ih = detector
    small = (
        cv2.resize(frame, (iw, ih))
        if (frame.shape[1], frame.shape[0]) != (iw, ih)
        else frame
    )
    _, faces = det.detect(small)
    if faces is None or len(faces) == 0:
        return None
    sx, sy = frame.shape[1] / iw, frame.shape[0] / ih
    best = None
    for f in faces:
        x, y, w, h = f[:4]
        area = w * h
        if best is None or area > best[3]:
            best = ((x + w / 2) * sx, (y + h / 2) * sy, w * sx, area * sx * sy)
    return best  # (cx, cy, size, area) o None


def _analyze(clip: Path, sample_every: float = SAMPLE_EVERY):
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise VideoError(f"no se pudo abrir: {clip}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = min(fps, FPS_CAP)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    step = max(1, int(round(fps * sample_every)))
    detector = _make_detector(w, h)
    detections = []
    btop, bbot, blft, brgt = [], [], [], []
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            g = frame.mean(axis=2)
            rm = g.max(axis=1)
            rows = np.where(rm > 14)[0]
            if len(rows):
                btop.append(rows.min())
                bbot.append(rows.max() + 1)
                cm = g.max(axis=0)
                cols = np.where(cm > 14)[0]
                blft.append(cols.min())
                brgt.append(cols.max() + 1)
            det = _detect_face_frame(frame, detector)
            t = idx / fps
            if det:
                detections.append(
                    {
                        "t": round(t, 3),
                        "cx": float(round(det[0], 1)),
                        "cy": float(round(det[1], 1)),
                        "size": float(round(det[2], 1)),
                    }
                )
        idx += 1
    cap.release()
    if n_frames <= 0:
        n_frames = idx
    bbox = None
    if len(btop) >= 8:
        l, t, r, b = (int(np.median(v)) for v in (blft, btop, brgt, bbot))
        bars = max(l, t, w - r, h - b)
        if bars >= 0.02 * min(w, h) and r > l and b > t:
            bbox = (l, t, r, b)
    return {
        "fps": fps,
        "width": w,
        "height": h,
        "n_frames": n_frames,
        "sample_step": step,
        "detections": detections,
        "bbox": bbox,
    }


def _build_path(analysis, aspect: float = 1.0):
    """Devuelve (path, crop_w, crop_h, axis, (l, t, r, b)).
    path = offset del eje con holgura por frame, sobre el área útil (sin letterbox).
    axis='x' -> path es x0; axis='y' -> path es y0."""
    fps, w, h = analysis["fps"], analysis["width"], analysis["height"]
    bb = analysis.get("bbox")
    l, t, r, b = bb if bb else (0, 0, w, h)
    cw, ch = r - l, b - t
    if aspect >= 1.0:
        crop_h = ch
        crop_w = min(cw, round(ch * aspect))
    else:
        crop_w = cw
        crop_h = min(ch, round(cw / aspect))
    x_slack, y_slack = cw - crop_w, ch - crop_h
    axis = "x" if x_slack >= y_slack else "y"
    neutral_main = (cw / 2) if axis == "x" else (ch / 2)
    limit = (cw - crop_w) if axis == "x" else (ch - crop_h)
    n = analysis["n_frames"]
    horizontal = axis == "x"

    target = np.full(n, np.nan, dtype=np.float64)
    for d in analysis["detections"]:
        i = min(n - 1, int(round(d["t"] * fps)))
        target[i] = (d["cx"] - l) if horizontal else (d["cy"] - t)

    # interpolación lineal entre detecciones; bordes -> detección más cercana; huecos -> neutral
    idx_known = np.where(~np.isnan(target))[0]
    if len(idx_known):
        first, last = idx_known[0], idx_known[-1]
        xs_known = np.interp(np.arange(first, last + 1), idx_known, target[idx_known])
        target[first : last + 1] = xs_known
        target[:first] = target[first]
        target[last + 1 :] = target[last]
    else:
        target[:] = neutral_main

    alpha = 1.0 - math.exp(-1.0 / (TAU * fps))
    path = np.empty(n, dtype=np.float64)
    path[0] = target[0] if not np.isnan(target[0]) else neutral_main
    for i in range(1, n):
        tv = target[i] if not np.isnan(target[i]) else neutral_main
        path[i] = path[i - 1] + alpha * (tv - path[i - 1])
    half = (crop_w if axis == "x" else crop_h) / 2.0
    path = np.clip(path, half, limit + half) - half  # offset del eje con holgura
    return path, crop_w, crop_h, axis, (l, t, r, b)


# ---------- generación de proyecto Shotcut (.mlt) ----------


def _fraction(fps: float) -> tuple[int, int]:
    best = (round(fps * 1000), 1000)
    for den in (1, 1001, 1000):
        num = round(fps * den)
        if abs(num / den - fps) < 1e-3:
            best = (num, den)
            break
    num, den = best
    while num % 2 == 0 and den % 2 == 0 and den > 1:
        num, den = num // 2, den // 2
    return num, den


def _build_mlt(
    mlt_path: Path,
    clip: Path,
    ranges: list[list[float]] | None,
    aspect: float,
    music: Path | None,
    music_db: float,
):
    """Escribe un .mlt estilo Shotcut con el mismo corte/encuadre del pipeline.

    Sin validación de render: abrir en Shotcut para revisar/retocar.
    El seguimiento de caras se vuelca como keyframes en transition.ox/oy.
    """
    import xml.etree.ElementTree as ET

    p = _check_file(str(clip), "clip")
    a = _analyze(p)  # timeline completo del fuente (los filtros viven en el source)
    path, crop_w, crop_h, axis, (x_off, y_off, x_end, y_end) = _build_path(a, aspect)
    fps = a["fps"]
    n_src = a["n_frames"]
    w, h = a["width"], a["height"]
    cw, ch = (x_end - x_off), (y_end - y_off)
    out_h = TARGET if aspect >= 1.0 else 2 * round(TARGET / aspect / 2)
    out_w = 2 * round(TARGET * aspect / 2) if aspect >= 1.0 else TARGET
    scale = out_h / ch if axis == "x" else out_w / cw

    if ranges:
        keep = [
            (
                int(round(_parse_time(r[0]) * fps)),
                int(round(_parse_time(r[1]) * fps)) - 1,
            )
            for r in ranges
        ]
    else:
        keep = [(0, n_src - 1)]
    keep = [(max(0, s), min(n_src - 1, e)) for s, e in keep if e >= s]
    total = sum(e - s + 1 for s, e in keep)

    mlt = ET.Element(
        "mlt",
        {
            "LC_NUMERIC": "C",
            "version": "7.0.0",
            "producer": "tractor0",
            "root": str(clip.parent),
        },
    )
    if aspect == 1.0 and abs(fps % 30) < 0.01:
        prof_desc = "Square 1080p 30 fps"
    elif aspect == 1.0:
        prof_desc = f"Square {fps:g} fps"
    else:
        prof_desc = f"Custom {out_w}x{out_h}"
    num, den = _fraction(fps)
    ET.SubElement(
        mlt,
        "profile",
        {
            "description": prof_desc,
            "width": str(out_w),
            "height": str(out_h),
            "progressive": "1",
            "sample_aspect_num": "1",
            "sample_aspect_den": "1",
            "display_aspect_num": str(out_w),
            "display_aspect_den": str(out_h),
            "colorspace": "709",
            "frame_rate_num": str(num),
            "frame_rate_den": str(den),
        },
    )

    chain = ET.SubElement(mlt, "chain", {"id": "chain0", "out": str(n_src - 1)})
    for k, v in [
        ("resource", str(clip)),
        ("audio_index", "0"),
        ("video_index", "0"),
        ("mlt_service", "avformat"),
        ("aspect_ratio", "1"),
        ("length", str(n_src)),
    ]:
        ET.SubElement(chain, "property", {"name": k}).text = v
    if (x_off, y_off, x_end, y_end) != (0, 0, w, h):
        f1 = ET.SubElement(chain, "filter", {"id": "filter_deletterbox"})
        ET.SubElement(f1, "property", {"name": "mlt_service"}).text = "crop"
        for name, val in [
            ("left", x_off),
            ("right", w - x_end),
            ("top", y_off),
            ("bottom", h - y_end),
        ]:
            if val > 0:
                ET.SubElement(f1, "property", {"name": name}).text = str(val)
        ET.SubElement(f1, "property", {"name": "active"}).text = "1"

    f2 = ET.SubElement(chain, "filter", {"id": "filter_transform"})
    ET.SubElement(f2, "property", {"name": "mlt_service"}).text = "affine"
    ET.SubElement(f2, "property", {"name": "transition.scale_x"}).text = f"{scale:.4f}"
    ET.SubElement(f2, "property", {"name": "transition.scale_y"}).text = f"{scale:.4f}"
    ET.SubElement(f2, "property", {"name": "transition.invert_scale"}).text = "1"
    axis_slack = (cw - crop_w) if axis == "x" else (ch - crop_h)
    if axis_slack > 1:
        step = max(1, int(round(fps * SAMPLE_EVERY)))
        vals = []
        for i in range(0, n_src, step):
            off = float(path[i])
            ox = round(-off * scale, 1)
            vals.append(f"{i}={ox}")
        if vals and (n_src - 1) % step != 0:
            vals.append(f"{n_src - 1}={round(-float(path[-1]) * scale, 1)}")
        prop = "transition.ox" if axis == "x" else "transition.oy"
        ET.SubElement(f2, "property", {"name": prop}).text = " ".join(vals)
    else:
        prop = "transition.ox" if axis == "x" else "transition.oy"
        ET.SubElement(f2, "property", {"name": prop}).text = "0"

    playlist = ET.SubElement(mlt, "playlist", {"id": "playlist0"})
    for s, e in keep:
        ET.SubElement(
            playlist, "entry", {"producer": "chain0", "in": str(s), "out": str(e)}
        )

    tractor_tracks = [("playlist0", "track0")]
    if music:
        m = _check_file(str(music), "música")
        m_info = _video_info(m)
        m_frames = int(round(m_info["duration"] * fps))
        mchain = ET.SubElement(
            mlt, "chain", {"id": "music_chain", "out": str(m_frames - 1)}
        )
        for k, v in [
            ("resource", str(m)),
            ("audio_index", "0"),
            ("video_index", "-1"),
            ("mlt_service", "avformat"),
        ]:
            ET.SubElement(mchain, "property", {"name": k}).text = v
        mf = ET.SubElement(mchain, "filter", {"id": "filter_music_volume"})
        ET.SubElement(mf, "property", {"name": "mlt_service"}).text = "volume"
        ET.SubElement(mf, "property", {"name": "level"}).text = str(float(music_db))
        mplaylist = ET.SubElement(mlt, "playlist", {"id": "music_playlist"})
        ET.SubElement(
            mplaylist,
            "entry",
            {
                "producer": "music_chain",
                "in": "0",
                "out": str(min(m_frames, total) - 1),
            },
        )
        tractor_tracks.append(("music_playlist", "track1"))

    tractor = ET.SubElement(
        mlt, "tractor", {"id": "tractor0", "in": "0", "out": str(total - 1)}
    )
    for pl, tid in tractor_tracks:
        ET.SubElement(tractor, "track", {"producer": pl, "id": tid})
    note = ET.SubElement(tractor, "property", {"name": "notes"})
    note.text = (
        "video-square-mcp: proyecto retocable. El loudnorm final lo aplica el "
        "export del MCP; la música va tal cual con level=%.1f dB" % music_db
    )

    tree = ET.ElementTree(mlt)
    ET.indent(tree, space="  ")
    tree.write(str(mlt_path), encoding="utf-8", xml_declaration=True)
    return mlt_path


# ---------- loudnorm en dos pasadas ----------


def _loudnorm_measure(
    src: Path,
    pre_filter: str | None,
    inputs_music: Path | None,
    music_db: float,
    dur: float,
):
    cmd = ["ffmpeg", "-hide_banner", "-i", str(src)]
    if inputs_music:
        cmd += ["-stream_loop", "-1", "-i", str(inputs_music)]
    fc = _mix_filter(
        pre_filter,
        inputs_music is not None,
        music_db,
        dur,
        "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
    )
    cmd += ["-filter_complex", fc, "-map", "[aout]", "-f", "null", "-"]
    proc = _run(cmd)
    stderr = proc.stderr or ""
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S)
    if not m:
        m = re.search(r"\{.*\}", stderr, re.S)
    if not m:
        raise VideoError("no se pudo medir loudnorm (¿video sin audio?)")
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        raise VideoError("salida loudnorm no parseable")
    return data


def _mix_filter(pre_filter, with_music, music_db, dur, final_loudnorm):
    parts = []
    if with_music:
        parts.append(f"[1:a]atrim=0:{dur:.3f},asetpts=N/SR/TB,volume={music_db}dB[m]")
        if pre_filter:
            parts.append(pre_filter.replace("[ain]", "[0:a]").replace("[amix]", "[a0]"))
            parts.append("[a0][m]amix=inputs=2:duration=first:normalize=0[mix]")
        else:
            parts.append("[0:a][m]amix=inputs=2:duration=first:normalize=0[mix]")
        parts.append(f"[mix]{final_loudnorm}[aout]")
    else:
        if pre_filter:
            parts.append(pre_filter.replace("[ain]", "[0:a]").replace("[amix]", "[a0]"))
            parts.append(f"[a0]{final_loudnorm}[aout]")
        else:
            parts.append(f"[0:a]{final_loudnorm}[aout]")
    return ";".join(parts)


def _loudnorm_apply_filter(meas):
    keys = {
        k: meas.get(k)
        for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    }
    return (
        f"loudnorm=I=-14:TP=-1.5:LRA=11:measured_I={keys['input_i']}:"
        f"measured_TP={keys['input_tp']}:measured_LRA={keys['input_lra']}:"
        f"measured_thresh={keys['input_thresh']}:offset={keys['target_offset']}:linear=true"
    )


# ---------- tools ----------


@mcp.tool
def probe(clip: str) -> dict:
    """Metadatos de un clip: duración, resolución, fps, codecs, audio."""
    p = _check_file(clip, "clip")
    return _video_info(p)


@mcp.tool
def detect_subjects(clip: str, sample_every: float = SAMPLE_EVERY) -> dict:
    """Detecta caras muestreando el clip. Devuelve trayectoria resumida y cobertura."""
    p = _check_file(clip, "clip")
    a = _analyze(p, float(sample_every))
    coverage = float(
        round(
            len(a["detections"])
            * a["sample_step"]
            / max(1e-6, a["n_frames"] / a["fps"]),
            3,
        )
    )
    return {
        "clip": str(p),
        "width": a["width"],
        "height": a["height"],
        "fps": a["fps"],
        "frames": a["n_frames"],
        "sampled": a["n_frames"] // a["sample_step"],
        "faces_detected": len(a["detections"]),
        "coverage": min(1.0, coverage),
        "deletterbox": list(a["bbox"]) if a.get("bbox") else None,
        "first": a["detections"][0] if a["detections"] else None,
        "last": a["detections"][-1] if a["detections"] else None,
    }


@mcp.tool
def reframe_square(clip: str, out: str, aspect: float = 1.0) -> dict:
    """Recorta a 1:1 (o aspect=ancho/alto, ej. 1.2 = 20% más ancho) y sigue al sujeto."""
    aspect = float(aspect)
    if not (0.4 <= aspect <= 2.5):
        raise VideoError("aspect fuera de rango razonable (0.4–2.5)")
    if abs(aspect - 1.0) < 1e-6:
        out_w = out_h = TARGET
    elif aspect >= 1.0:
        out_h = TARGET
        out_w = 2 * round(TARGET * aspect / 2)
    else:
        out_w = TARGET
        out_h = 2 * round(TARGET / aspect / 2)
    p = _check_file(clip, "clip")
    out_p = Path(out).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    a = _analyze(p)
    path, crop_w, crop_h, axis, (x_off, y_off, x_end, y_end) = _build_path(a, aspect)
    cap = cv2.VideoCapture(str(p))
    n_target = a["n_frames"]
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{out_w}x{out_h}",
        "-r",
        f"{a['fps']:.6f}",
        "-i",
        "-",
        "-i",
        str(p),
        "-map",
        "0:v",
        "-map",
        "1:a?",
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-shortest",
        str(out_p),
    ]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
    )
    i = 0
    try:
        while i < n_target:
            ok, frame = cap.read()
            if not ok:
                break
            off = int(path[min(i, len(path) - 1)])
            crop = frame[y_off:y_end, x_off:x_end]
            if axis == "x":
                x0, y0 = off, 0
            else:
                x0, y0 = 0, off
            sq = crop[y0 : y0 + crop_h, x0 : x0 + crop_w]
            if sq.shape[0] != out_h or sq.shape[1] != out_w:
                sq = cv2.resize(sq, (out_w, out_h), interpolation=cv2.INTER_AREA)
            proc.stdin.write(sq.tobytes())
            i += 1
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
    stderr = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
    if proc.wait() != 0:
        raise VideoError(f"render falló:\n{stderr[-3000:]}")
    coverage = float(
        round(
            len(a["detections"]) * a["sample_step"] / max(1e-6, n_target / a["fps"]), 3
        )
    )
    return {
        "ok": True,
        "out": str(out_p),
        "frames": i,
        "subject_coverage": min(1.0, coverage),
        "deletterbox": [x_off, y_off, x_end, y_end] if a.get("bbox") else None,
        "aspect": aspect,
        "size": f"{out_w}x{out_h}",
        "note": "sin caras detectadas; encuadre centrado"
        if not a["detections"]
        else "ok",
    }


@mcp.tool
def cut(clip: str, ranges: list[list], out: str) -> dict:
    """Corta conservando [inicio, fin] en segundos. Acepta 'MM:SS' o segundos."""
    p = _check_file(clip, "clip")
    if not ranges:
        raise VideoError("ranges vacío: pasa [[inicio, fin], ...]")
    parsed = []
    for r in ranges:
        if len(r) != 2:
            raise VideoError(f"cada range es [inicio, fin], recibí: {r}")
        s, e = _parse_time(r[0]), _parse_time(r[1])
        if e <= s:
            raise VideoError(f"range con fin <= inicio: {r}")
        parsed.append((s, e))
    out_p = Path(out).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vsq_cut_") as tmp:
        segs = []
        for i, (s, e) in enumerate(parsed):
            seg = Path(tmp) / f"seg_{i:03d}.mp4"
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    f"{s:.3f}",
                    "-to",
                    f"{e:.3f}",
                    "-i",
                    str(p),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    str(seg),
                ]
            )
            segs.append(seg)
        listfile = Path(tmp) / "concat.txt"
        listfile.write_text("".join(f"file '{s}'\n" for s in segs))
        _run(
            [
                "ffmpeg",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(listfile),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(out_p),
            ]
        )
    return {"ok": True, "out": str(out_p), "segments": len(parsed)}


@mcp.tool
def concat(clips: list[str], out: str) -> dict:
    """Une varios clips (misma spec recomendada) en uno solo."""
    paths = [_check_file(c, "clip") for c in clips]
    if len(paths) < 2:
        raise VideoError("pasa al menos 2 clips")
    out_p = Path(out).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vsq_cat_") as tmp:
        listfile = Path(tmp) / "concat.txt"
        listfile.write_text("".join(f"file '{p}'\n" for p in paths))
        try:
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listfile),
                    "-c",
                    "copy",
                    "-movflags",
                    "+faststart",
                    str(out_p),
                ]
            )
        except VideoError:
            # fallback: re-encode
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(listfile),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-preset",
                    "medium",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    str(out_p),
                ]
            )
    return {"ok": True, "out": str(out_p), "clips": len(paths)}


@mcp.tool
def add_music(video: str, music: str, out: str, music_db: float = -18.0) -> dict:
    """Mezcla música de fondo bajo el audio original; loudnorm a -14 LUFS."""
    v = _check_file(video, "video")
    m = _check_file(music, "música")
    out_p = Path(out).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    info = _video_info(v)
    dur = max(info["duration"], 0.1)
    if info["has_audio"]:
        meas = _loudnorm_measure(v, None, m, music_db, dur)
        apply_ln = _loudnorm_apply_filter(meas)
        fc = _mix_filter(None, True, music_db, dur, apply_ln)
    else:
        ln = "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json"
        probe_cmd = [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(m),
            "-filter_complex",
            f"[0:a]atrim=0:{dur:.3f},asetpts=N/SR/TB,volume={music_db}dB,{ln}[aout]",
            "-map",
            "[aout]",
            "-f",
            "null",
            "-",
        ]
        proc = _run(probe_cmd)
        meas = json.loads(
            re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", proc.stderr or "", re.S).group(0)
        )
        apply_ln = _loudnorm_apply_filter(meas)
        fc = f"[1:a]atrim=0:{dur:.3f},asetpts=N/SR/TB,volume={music_db}dB,{apply_ln}[aout]"
    _run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(v),
            "-stream_loop",
            "-1",
            "-i",
            str(m),
            "-filter_complex",
            fc,
            "-map",
            "0:v",
            "-map",
            "[aout]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            str(out_p),
        ]
    )
    return {
        "ok": True,
        "out": str(out_p),
        "music_db": music_db,
        "lufs_target": -14,
        "duration": dur,
    }


@mcp.tool
def export(video: str, out: str) -> dict:
    """Exporta al spec final: H.264 yuv420p, AAC 192k, faststart, -14 LUFS.
    Conserva la resolución del input (solo fuerza dimensiones pares y SAR 1:1)."""
    v = _check_file(video, "video")
    out_p = Path(out).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    info = _video_info(v)
    dur = max(info["duration"], 0.1)
    vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2,setsar=1"
    if info["has_audio"]:
        meas = _loudnorm_measure(v, None, None, 0.0, dur)
        ln = _loudnorm_apply_filter(meas)
        af = f"[0:a]{ln}[aout]"
        fc = af
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(v),
            "-vf",
            vf,
            "-filter_complex",
            fc,
            "-map",
            "0:v",
            "-map",
            "[aout]",
        ]
    else:
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(v),
            "-vf",
            vf,
        ]
    cmd += [
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(out_p),
    ]
    _run(cmd)
    return {
        "ok": True,
        "out": str(out_p),
        "spec": "h264 yuv420p aac192k faststart -14LUFS (resolución conservada)",
    }


@mcp.tool
def pipeline(
    clip: str,
    out: str,
    ranges: list[list] | None = None,
    music: str | None = None,
    music_db: float = -18.0,
    aspect: float = 1.0,
    project: bool = True,
) -> dict:
    """Atajo: cut opcional -> reframe (aspect 1.0 = cuadrado, 1.2 = 20% más ancho)
    -> música opcional -> export final. Si project=true, además escribe un .mlt
    retocable en Shotcut (mismo corte/encuadre/materia) junto al MP4."""
    p = _check_file(clip, "clip")
    if music:
        _check_file(music, "música")
    out_p = Path(out).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    steps = []
    with tempfile.TemporaryDirectory(prefix="vsq_pipe_") as tmp:
        current = p
        if ranges:
            cut_out = Path(tmp) / "cut.mp4"
            cut(str(current), ranges, str(cut_out))
            current = cut_out
            steps.append("cut")
        ref_out = Path(tmp) / "reframe.mp4"
        reframe_square(str(current), str(ref_out), aspect)
        current = ref_out
        steps.append(f"reframe_square(aspect={aspect})")
        if music:
            mus_out = Path(tmp) / "music.mp4"
            add_music(str(current), music, str(mus_out), music_db)
            current = mus_out
            steps.append(f"add_music({music_db}dB)")
        export(str(current), str(out_p))
        steps.append("export")
    result = {"ok": True, "out": str(out_p), "steps": steps}
    if project:
        mlt_path = out_p.with_suffix(".mlt")
        try:
            _build_mlt(
                mlt_path,
                p,
                ranges,
                float(aspect),
                Path(music).expanduser().resolve() if music else None,
                float(music_db),
            )
            result["mlt"] = str(mlt_path)
            steps.append("project_mlt")
        except Exception as exc:  # el .mlt no debe tumbar el export ya hecho
            result["mlt"] = None
            result["mlt_error"] = str(exc)
    return result


if __name__ == "__main__":
    mcp.run()
