import type { CVRecord, QueryResponse } from "../types/cv";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";

export async function uploadCv(file: File): Promise<CVRecord> {
  const formData = new FormData();
  formData.append("file", file);
  const response = await fetch(`${API_BASE_URL}/api/cvs/upload`, {
    method: "POST",
    body: formData,
  });
  if (!response.ok) {
    throw new Error(`Upload failed with status ${response.status}`);
  }
  return response.json();
}

export async function queryCvs(question: string, topK = 5): Promise<QueryResponse> {
  const response = await fetch(`${API_BASE_URL}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });
  if (!response.ok) {
    throw new Error(`Query failed with status ${response.status}`);
  }
  return response.json();
}

export function getCvFileUrl(cvId: string): string {
  return `${API_BASE_URL}/api/cvs/${cvId}/file`;
}
