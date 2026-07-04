import { useState } from "react";
import type { QueryResponse, RetrievedCV } from "../types/cv";
import { getCvFileUrl } from "../api/client";

interface ResultsViewProps {
  result: QueryResponse;
}

function CandidateCard({
  candidate,
  isBest,
  rank,
  reason,
}: {
  candidate: RetrievedCV;
  isBest: boolean;
  rank: number;
  reason: string | null;
}) {
  const [selectedExtraField, setSelectedExtraField] = useState("");
  const extraFieldNames = Object.keys(candidate.extraction.extra_fields);

  return (
    <div className={isBest ? "candidate-card candidate-card--best" : "candidate-card"}>
      {isBest && (
        <span className="stamp" aria-hidden="true">
          AI Pick
        </span>
      )}
      <span className="candidate-card__rank">#{rank}</span>
      <span className="candidate-card__tab">{candidate.extraction.domain ?? "Uncategorized"}</span>
      <h3 className="candidate-card__name">
        {candidate.extraction.name ?? candidate.filename}
        {isBest ? " (AI Pick)" : ""}
      </h3>
      {isBest && reason && <p className="candidate-card__reason">{reason}</p>}
      <p className="candidate-card__meta">
        {candidate.filename} &middot;{" "}
        <span
          className="candidate-card__score"
          title="Combined ranking score from hybrid keyword + semantic search. It reflects relative order among these results, not an absolute match percentage."
        >
          match score {(candidate.score * 100).toFixed(2)}%
        </span>
      </p>
      <a href={getCvFileUrl(candidate.id)} target="_blank" rel="noreferrer">
        View original CV
      </a>
      {candidate.extraction.skills.length > 0 && (
        <div className="tag-row">
          {candidate.extraction.skills.map((skill) => (
            <span key={skill} className="tag">
              {skill}
            </span>
          ))}
        </div>
      )}
      <pre data-testid={`cv-json-${candidate.id}`}>{JSON.stringify(candidate.extraction, null, 2)}</pre>
      {extraFieldNames.length > 0 && (
        <div className="extra-fields">
          <label htmlFor={`extra-fields-${candidate.id}`}>Additional fields found by AI</label>
          <select
            id={`extra-fields-${candidate.id}`}
            aria-label="Additional fields found by AI"
            value={selectedExtraField}
            onChange={(event) => setSelectedExtraField(event.target.value)}
          >
            <option value="">Select a field...</option>
            {extraFieldNames.map((fieldName) => (
              <option key={fieldName} value={fieldName}>
                {fieldName}
              </option>
            ))}
          </select>
          {selectedExtraField && <p>{candidate.extraction.extra_fields[selectedExtraField]}</p>}
        </div>
      )}
    </div>
  );
}

function ResultsView({ result }: ResultsViewProps) {
  return (
    <div className="results-view">
      <section>
        <p className="section-eyebrow">Assistant readout</p>
        <h2>Answer</h2>
        <div className="answer-panel">
          <p>{result.answer}</p>
        </div>
      </section>
      <section>
        <p className="section-eyebrow">{result.retrieved.length} match{result.retrieved.length === 1 ? "" : "es"} filed</p>
        <h2>Retrieved Candidates</h2>
        <div className="candidate-grid">
          {result.retrieved.map((candidate, index) => (
            <CandidateCard
              key={candidate.id}
              candidate={candidate}
              isBest={candidate.id === result.best_candidate_id}
              rank={index + 1}
              reason={candidate.id === result.best_candidate_id ? result.best_candidate_reason : null}
            />
          ))}
        </div>
      </section>
    </div>
  );
}

export default ResultsView;
