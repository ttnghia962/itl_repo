import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import UploadPanel from "./UploadPanel";
import * as client from "../api/client";

describe("UploadPanel", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("calls onUploaded with the uploaded record after a successful upload", async () => {
    const fakeRecord = { id: "cv1", filename: "alice.pdf", extraction: {}, raw_text: "" };
    vi.spyOn(client, "uploadCv").mockResolvedValue(fakeRecord as any);
    const onUploaded = vi.fn();

    render(<UploadPanel onUploaded={onUploaded} />);
    const file = new File(["dummy"], "alice.pdf", { type: "application/pdf" });
    const input = screen.getByLabelText("Choose CV PDF file") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [file] } });
    fireEvent.click(screen.getByText("Upload CV"));

    await waitFor(() => expect(onUploaded).toHaveBeenCalledWith(fakeRecord));
  });

  it("shows an error message when no file is selected", () => {
    render(<UploadPanel onUploaded={vi.fn()} />);
    fireEvent.click(screen.getByText("Upload CV"));
    expect(screen.getByRole("alert")).toHaveTextContent("Please choose a PDF file first.");
  });

  it("shows an error message when the upload fails", async () => {
    vi.spyOn(client, "uploadCv").mockRejectedValue(new Error("Upload failed with status 500"));
    render(<UploadPanel onUploaded={vi.fn()} />);
    const file = new File(["dummy"], "alice.pdf", { type: "application/pdf" });
    const input = screen.getByLabelText("Choose CV PDF file") as HTMLInputElement;
    fireEvent.change(input, { target: { files: [file] } });
    fireEvent.click(screen.getByText("Upload CV"));

    await waitFor(() =>
      expect(screen.getByRole("alert")).toHaveTextContent("Upload failed with status 500")
    );
  });
});
