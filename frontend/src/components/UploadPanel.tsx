import { useRef, useState } from "react";
import { uploadCv } from "../api/client";
import type { CVRecord } from "../types/cv";

interface UploadPanelProps {
  onUploaded: (record: CVRecord) => void;
}

function UploadPanel({ onUploaded }: UploadPanelProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleUploadClick() {
    const file = fileInputRef.current?.files?.[0];
    if (!file) {
      setError("Please choose a PDF file first.");
      return;
    }
    setError(null);
    setIsUploading(true);
    try {
      const record = await uploadCv(file);
      onUploaded(record);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setIsUploading(false);
    }
  }

  return (
    <div className="upload-panel">
      <p className="panel-label">Candidate intake</p>
      <div className="upload-panel__row">
        <input ref={fileInputRef} type="file" accept="application/pdf" aria-label="Choose CV PDF file" />
        <button onClick={handleUploadClick} disabled={isUploading}>
          {isUploading ? "Uploading..." : "Upload CV"}
        </button>
      </div>
      {error && (
        <p role="alert">
          <strong>Error</strong>
          {error}
        </p>
      )}
    </div>
  );
}

export default UploadPanel;
