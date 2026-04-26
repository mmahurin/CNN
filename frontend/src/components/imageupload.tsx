import { useRef } from "react"
import axios from "axios"
import type { Model, Detection } from "../App"

interface Props {
  model: Model
  onResult: (result: Detection[] | Detection) => void
  onImageUrl: (url: string) => void
  setLoading: (loading: boolean) => void
}

export default function ImageUpload({ model, onResult, onImageUrl, setLoading }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  async function handleFile(file: File) {
    onImageUrl(URL.createObjectURL(file))
    setLoading(true)

    const form = new FormData()
    form.append("file", file)

    try {
      const { data } = await axios.post(
        `http://127.0.0.1:8000/detect/${model}`,
        form,
        { headers: { "Content-Type": "multipart/form-data" } }
      )
      onResult(data)
    } catch (err) {
      alert("Detection failed. Make sure the FastAPI server is running.")
    } finally {
      setLoading(false)
    }
  }

  function handleDrop(e: React.DragEvent) {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  return (
    <div
      onDrop={handleDrop}
      onDragOver={(e) => e.preventDefault()}
      onClick={() => inputRef.current?.click()}
      style={{
        border: "2px dashed #646cff",
        borderRadius: 12,
        padding: 48,
        textAlign: "center",
        cursor: "pointer",
        marginBottom: 24,
        color: "#aaa"
      }}
    >
      <p style={{ margin: 0 }}>Drag & drop an image here, or click to upload</p>
      <input
        ref={inputRef}
        type="file"
        accept="image/*"
        style={{ display: "none" }}
        onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
      />
    </div>
  )
}
