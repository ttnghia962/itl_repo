import { describe, it, expect, vi, afterEach } from "vitest";
import { uploadCv, queryCvs, getCvFileUrl } from "./client";

describe("api client", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("uploadCv posts multipart form data and returns parsed JSON", async () => {
    const fakeResponse = { id: "cv1", filename: "alice.pdf", extraction: {}, raw_text: "" };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => fakeResponse,
    });
    vi.stubGlobal("fetch", fetchMock);

    const file = new File(["dummy"], "alice.pdf", { type: "application/pdf" });
    const result = await uploadCv(file);

    expect(result).toEqual(fakeResponse);
    expect(fetchMock).toHaveBeenCalledWith(
      expect.stringContaining("/api/cvs/upload"),
      expect.objectContaining({ method: "POST" })
    );
  });

  it("queryCvs posts the question and returns the parsed answer", async () => {
    const fakeResponse = { answer: "Alice fits best", best_candidate_id: "cv1", retrieved: [] };
    const fetchMock = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => fakeResponse,
    });
    vi.stubGlobal("fetch", fetchMock);

    const result = await queryCvs("Who knows Python?", 3);

    expect(result).toEqual(fakeResponse);
    const [, options] = fetchMock.mock.calls[0];
    expect(JSON.parse(options.body)).toEqual({ question: "Who knows Python?", top_k: 3 });
  });

  it("getCvFileUrl builds the pdf file url for a candidate id", () => {
    expect(getCvFileUrl("cv1")).toContain("/api/cvs/cv1/file");
  });

  it("uploadCv throws when the response is not ok", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: false, status: 500 }));
    const file = new File(["dummy"], "alice.pdf");
    await expect(uploadCv(file)).rejects.toThrow("Upload failed with status 500");
  });
});
