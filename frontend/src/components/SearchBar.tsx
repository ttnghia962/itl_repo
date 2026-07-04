import { useState } from "react";
import { queryCvs } from "../api/client";
import type { QueryResponse } from "../types/cv";

interface SearchBarProps {
  onResult: (result: QueryResponse) => void;
}

function SearchBar({ onResult }: SearchBarProps) {
  const [question, setQuestion] = useState("");
  const [isSearching, setIsSearching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSearch() {
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }
    setError(null);
    setIsSearching(true);
    try {
      const result = await queryCvs(question);
      onResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed.");
    } finally {
      setIsSearching(false);
    }
  }

  return (
    <div className="search-bar">
      <p className="panel-label">Ask the assistant</p>
      <div className="search-bar__row">
        <input
          type="text"
          aria-label="Ask a question about candidates"
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          placeholder="e.g. Find a candidate with Python and Machine Learning experience"
        />
        <button onClick={handleSearch} disabled={isSearching}>
          {isSearching ? "Searching..." : "Search"}
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

export default SearchBar;
