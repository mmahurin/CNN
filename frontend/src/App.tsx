import { useState } from "react"
import ImageUpload from "./components/imageupload"
import DetectionResult from "./components/detectionresults"

export type Model = "binary" | "multiclass" | "rcnn"

export interface Detection {
  label: string
  confidence: number
  box?: { xmin: number; ymin: number; xmax: number; ymax: number }
  all_scores?: Record<string, number>
}

function App() {
  const [model, setModel] = useState<Model>("rcnn")
  const [result, setResult] = useState<Detection[] | Detection | null>(null)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  return (
    <div style={{ maxWidth: 900, margin: "0 auto", padding: 32, fontFamily: "sans-serif" }}>
      <h1>Vehicle Classifier</h1>

      <div style={{ marginBottom: 24 }}>
        <label style={{ marginRight: 12 }}>Select Model:</label>
        {(["binary", "multiclass", "rcnn"] as Model[]).map((m) => (
          <button
            key={m}
            onClick={() => { setModel(m); setResult(null) }}
            style={{
              marginRight: 8,
              padding: "6px 16px",
              background: model === m ? "#646cff" : "#1a1a1a",
              color: "white",
              border: "1px solid #646cff",
              borderRadius: 6,
              cursor: "pointer"
            }}
          >
            {m === "binary" ? "Binary CNN" : m === "multiclass" ? "Multi-class CNN" : "Faster R-CNN"}
          </button>
        ))}
      </div>

      <ImageUpload
        model={model}
        onResult={setResult}
        onImageUrl={setImageUrl}
        setLoading={setLoading}
      />

      {loading && <p>Running detection...</p>}

      {result && imageUrl && (
        <DetectionResult result={result} imageUrl={imageUrl} model={model} />
      )}
    </div>
  )
}

export default App
