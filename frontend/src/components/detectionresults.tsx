import { useRef, useEffect } from "react"
import type { Model, Detection } from "../App"

interface Props {
  result: Detection[] | Detection
  imageUrl: string
  model: Model
}

export default function DetectionResult({ result, imageUrl, model }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const img = new Image()
    img.src = imageUrl
    img.onload = () => {
      canvas.width  = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)

      if (model === "rcnn" && Array.isArray(result)) {
        result.forEach(({ label, confidence, box }) => {
          if (!box) return
          const x = box.xmin * img.width
          const y = box.ymin * img.height
          const w = (box.xmax - box.xmin) * img.width
          const h = (box.ymax - box.ymin) * img.height

          ctx.strokeStyle = "#646cff"
          ctx.lineWidth   = 3
          ctx.strokeRect(x, y, w, h)

          ctx.fillStyle = "#646cff"
          ctx.fillRect(x, y - 24, w, 24)
          ctx.fillStyle = "white"
          ctx.font      = "bold 14px sans-serif"
          ctx.fillText(`${label} ${Math.round(confidence * 100)}%`, x + 4, y - 6)
        })
      }
    }
  }, [result, imageUrl, model])

  const single = !Array.isArray(result) ? result as Detection : null

  return (
    <div>
      <canvas
        ref={canvasRef}
        style={{ maxWidth: "100%", borderRadius: 8, display: "block", marginBottom: 16 }}
      />

      {single && (
        <div style={{ background: "#1a1a1a", padding: 16, borderRadius: 8 }}>
          <p style={{ margin: 0 }}>
            Prediction: <strong>{single.label}</strong> —{" "}
            {Math.round(single.confidence * 100)}% confidence
          </p>
          {single.all_scores && (
            <ul style={{ marginTop: 8 }}>
              {Object.entries(single.all_scores).map(([cls, score]) => (
                <li key={cls}>{cls}: {Math.round((score as number) * 100)}%</li>
              ))}
            </ul>
          )}
        </div>
      )}

      {Array.isArray(result) && result.length === 0 && (
        <p>No vehicles detected above 50% confidence.</p>
      )}
    </div>
  )
}
