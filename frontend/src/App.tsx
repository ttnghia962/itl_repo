import { useState } from "react";
import UploadPanel from "./components/UploadPanel";
import SearchBar from "./components/SearchBar";
import ResultsView from "./components/ResultsView";
import type { CVRecord, QueryResponse } from "./types/cv";

function App() {
  const [lastUploaded, setLastUploaded] = useState<CVRecord | null>(null);
  const [queryResult, setQueryResult] = useState<QueryResponse | null>(null);

  return (
    <div className="app">
      <header className="masthead">
        <p className="masthead__eyebrow">Candidate file room</p>
        <h1>CV RAG Assistant</h1>
        <span className="masthead__stamp stamp" aria-hidden="true">
          AI Reviewed
        </span>
      </header>

      <div className="intake">
        <UploadPanel onUploaded={setLastUploaded} />
        <SearchBar onResult={setQueryResult} />
      </div>

      {lastUploaded && (
        <p className="receipt">
          Filed: <strong>{lastUploaded.filename}</strong> (id: {lastUploaded.id})
        </p>
      )}

      {queryResult ? (
        <ResultsView result={queryResult} />
      ) : (
        <div className="empty-state">
          <p>Upload a résumé, then ask a question to see AI-matched candidates here.</p>
        </div>
      )}
    </div>
  );
}

export default App;
