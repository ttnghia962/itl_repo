import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ResultsView from "./ResultsView";
import type { QueryResponse } from "../types/cv";

const sampleResult: QueryResponse = {
  answer: "Alice is the best fit for Python and Machine Learning.",
  best_candidate_id: "cv1",
  best_candidate_reason: "Alice has 6 years of hands-on Python and Machine Learning experience.",
  retrieved: [
    {
      id: "cv1",
      filename: "alice.pdf",
      score: 0.92,
      extraction: {
        name: "Alice",
        email: "alice@example.com",
        phone: "555-0100",
        skills: ["Python", "Machine Learning"],
        education: [],
        experience: [],
        current_role: "ML Engineer",
        domain: "Information Technology",
        extra_fields: { certifications: "AWS Certified Solutions Architect" },
      },
    },
  ],
};

const multiResult: QueryResponse = {
  answer: sampleResult.answer,
  best_candidate_id: "cv1",
  best_candidate_reason: sampleResult.best_candidate_reason,
  retrieved: [
    sampleResult.retrieved[0],
    {
      id: "cv2",
      filename: "bob.pdf",
      score: 0.31,
      extraction: {
        name: "Bob",
        email: "bob@example.com",
        phone: "555-0101",
        skills: ["Java"],
        education: [],
        experience: [],
        current_role: "Backend Engineer",
        domain: "Information Technology",
        extra_fields: {},
      },
    },
  ],
};

const noReasonResult: QueryResponse = {
  answer: sampleResult.answer,
  best_candidate_id: "cv1",
  best_candidate_reason: null,
  retrieved: sampleResult.retrieved,
};

describe("ResultsView", () => {
  it("renders the LLM answer and marks the AI pick", () => {
    render(<ResultsView result={sampleResult} />);
    expect(screen.getByText(sampleResult.answer)).toBeInTheDocument();
    expect(screen.getByText("Alice (AI Pick)")).toBeInTheDocument();
    expect(screen.getByText("AI Pick")).toBeInTheDocument();
  });

  it("renders extracted JSON for each candidate", () => {
    render(<ResultsView result={sampleResult} />);
    const json = screen.getByTestId("cv-json-cv1");
    expect(json.textContent).toContain('"Python"');
  });

  it("shows a dropdown of AI-discovered extra fields and reveals the value on selection", () => {
    render(<ResultsView result={sampleResult} />);
    const select = screen.getByLabelText("Additional fields found by AI");
    fireEvent.change(select, { target: { value: "certifications" } });
    expect(screen.getByText("AWS Certified Solutions Architect")).toBeInTheDocument();
  });

  it("links to the original CV file", () => {
    render(<ResultsView result={sampleResult} />);
    const link = screen.getByText("View original CV") as HTMLAnchorElement;
    expect(link.href).toContain("/api/cvs/cv1/file");
  });

  it("shows the rank position of every candidate in list order", () => {
    render(<ResultsView result={multiResult} />);
    expect(screen.getByText("#1")).toBeInTheDocument();
    expect(screen.getByText("#2")).toBeInTheDocument();
  });

  it("explains that the match score is a relative hybrid ranking score, not an absolute percentage", () => {
    render(<ResultsView result={sampleResult} />);
    const scoreEl = screen.getByText(/match score 92\.00%/);
    expect(scoreEl).toHaveAttribute(
      "title",
      "Combined ranking score from hybrid keyword + semantic search. It reflects relative order among these results, not an absolute match percentage."
    );
  });

  it("shows the AI's reason for its pick on the picked candidate's card", () => {
    render(<ResultsView result={sampleResult} />);
    expect(
      screen.getByText("Alice has 6 years of hands-on Python and Machine Learning experience.")
    ).toBeInTheDocument();
  });

  it("renders no reason box when best_candidate_reason is null", () => {
    render(<ResultsView result={noReasonResult} />);
    expect(screen.getByText("AI Pick")).toBeInTheDocument();
    expect(
      screen.queryByText("Alice has 6 years of hands-on Python and Machine Learning experience.")
    ).not.toBeInTheDocument();
  });
});
