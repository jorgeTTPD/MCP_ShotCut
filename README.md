# video-square-mcp

MCP local para convertir clips a **cuadrado 1080x1080** siguiendo al sujeto (detección de caras con YuNet/OpenCV), cortar por timecodes, unir clips, mezclar música de fondo y exportar para redes — todo sobre FFmpeg.

## Instalación

```bash
cd ~/video-square-mcp
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

El modelo YuNet ya está en `models/face_detection_yunet_2023mar.onnx`.

## Registro en opencode (global, `~/.config/opencode/opencode.jsonc`)

```jsonc
"video-square": {
  "type": "local",
  "command": ["/home/jorge/video-square-mcp/.venv/bin/python",
              "/home/jorge/video-square-mcp/video_square_mcp.py"],
  "enabled": true
}
```

## Tools

| Tool | Descripción |
|---|---|
| `probe(clip)` | Metadatos: duración, resolución, fps, codecs, audio |
| `detect_subjects(clip, sample_every=0.5)` | Trayectoria de caras + cobertura |
| `reframe_square(clip, out)` | 1080x1080, crop que sigue al sujeto; **auto-deletterbox** (detecta barras embebidas y recorta antes del crop) |
| `cut(clip, ranges, out)` | Conserva `[[ini, fin], ...]`; segundos o `"MM:SS"` |
| `concat(clips, out)` | Une clips (re-encode si no son compatibles) |
| `add_music(video, music, out, music_db=-18)` | Música bajo el audio original, loudnorm -14 LUFS |
| `export(video, out)` | Spec final: H.264 yuv420p, AAC 192k, faststart, -14 LUFS |
| `pipeline(clip, out, ranges?, music?, music_db=-18, aspect=1.0, project=true)` | cut → reframe → música → export; **project=true escribe también un `.mlt`** junto al MP4 |

## Ejemplo de uso

> "Pasa clip1.mp4 y clip2.mp4 a cuadrado con música de fondo.mp3, conserva de clip1 solo 0:05–0:40"

## Proyecto Shotcut (.mlt)

`pipeline(..., project=true)` (default) escribe `<salida>.mlt` junto al MP4: mismo fuente, mismo corte (entry in/out), deletterbox como filtro `crop` estático, encuadre con keyframes de caras en `transition.ox`/`transition.oy` del filtro `affine`, y la música en una segunda pista con `volume` en dB. Es para **retocar a mano** en Shotcut; el loudnorm final solo lo aplica el export del MCP. El `.mlt` no se autovalida renderizando — abrirlo en Shotcut para revisar.

## Verificación técnica

- Loudness integrado verificado con `ebur128` → **-14.0 LUFS**
- Reframe validado con clip sintético (cara en movimiento): cobertura 100 %, sujeto siempre en cuadro
- `test_pipeline.py <clip>` corre el flujo completo contra un cliente MCP en memoria

## Notas

- El crop sigue la cara con suavizado EMA (tau ≈ 0.8 s); si no hay cara detectada, encuadra al centro.
- YuNet detecta caras frontales/de perfil cercanas; cuerpos completos quedan como mejora futura.
- **Auto-deletterbox:** si el video trae barras negras embebidas (p. ej. 2.39:1 en un 480p), se detectan por mediana de frames y se recorta antes del crop cuadrado. Desactivarlo no es opción aún; se puede pedir.
- El video se re-codifica siempre (H.264 CRF 18, preset medium).
- Tras editar `video_square_mcp.py`, recargar los MCP en opencode para que el servidor stdio use el código nuevo.
