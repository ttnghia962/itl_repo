import { describe, it, expect, vi, afterEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import App from "./App";
import * as client from "./api/client";

describe("App", () => {
  afterEach(() => vi.restoreAllMocks());

  it("shows query results after a search", async () => {
    vi.spyOn(client, "queryCvs").mockResolvedValue({
      answer: "Alice fits best.",
      best_candidate_id: "cv1",
      best_candidate_reason: "Alice has the strongest Python background.",
      retrieved: [
        {
          id: "cv1",
          filename: "alice.pdf",
          score: 0.9,
          extraction: {
            name: "Alice",
            email: null,
            phone: null,
            skills: ["Python"],
            education: [],
            experience: [],
            current_role: "ML Engineer",
            domain: "Information Technology",
            extra_fields: {},
          },
        },
      ],
    });

    render(<App />);
    fireEvent.change(screen.getByLabelText("Ask a question about candidates"), {
      target: { value: "Who knows Python?" },
    });
    fireEvent.click(screen.getByText("Search"));

    await waitFor(() => expect(screen.getByText("Alice fits best.")).toBeInTheDocument());
  });
});
