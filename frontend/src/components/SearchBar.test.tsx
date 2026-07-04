import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import SearchBar from "./SearchBar";
import * as client from "../api/client";

describe("SearchBar", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("calls onResult with the query response after a successful search", async () => {
    const fakeResult = { answer: "Alice fits best", best_candidate_id: "cv1", retrieved: [] };
    vi.spyOn(client, "queryCvs").mockResolvedValue(fakeResult as any);
    const onResult = vi.fn();

    render(<SearchBar onResult={onResult} />);
    fireEvent.change(screen.getByLabelText("Ask a question about candidates"), {
      target: { value: "Who knows Python?" },
    });
    fireEvent.click(screen.getByText("Search"));

    await waitFor(() => expect(onResult).toHaveBeenCalledWith(fakeResult));
  });

  it("shows an error message when the question is empty", () => {
    render(<SearchBar onResult={vi.fn()} />);
    fireEvent.click(screen.getByText("Search"));
    expect(screen.getByRole("alert")).toHaveTextContent("Please enter a question.");
  });
});
