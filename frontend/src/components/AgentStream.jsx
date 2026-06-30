/**
 * LEARN: AgentStream manages the live SSE connection to the backend.
 *
 * How streaming works here:
 * 1. User submits a question
 * 2. We call fetch() — this starts the HTTP request but doesn't wait for it to finish
 * 3. response.body is a ReadableStream — we read it chunk by chunk as it arrives
 * 4. Each chunk is raw bytes → we decode to text → parse SSE format → extract JSON
 * 5. Each parsed event gets added to our `events` state array → React re-renders
 *
 * Why not EventSource? EventSource only works with GET requests. We need POST
 * to send the question in the request body.
 */
import { useState, useRef } from "react";
import ToolCallBadge from "./ToolCallBadge";
import FinalAnswer from "./FinalAnswer";

const API_URL = "http://localhost:8000";

export default function AgentStream() {
  const [question, setQuestion] = useState("");
  const [events, setEvents] = useState([]);      // all agent steps so far
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState(null);
  const bottomRef = useRef(null);                 // used to auto-scroll

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim() || isRunning) return;

    setEvents([]);
    setError(null);
    setIsRunning(true);

    try {
      // Start the streaming POST request
      const response = await fetch(`${API_URL}/research`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      // response.body is a ReadableStream — getReader() lets us pull chunks
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      // Tool calls accumulate here so we can pair them with their results
      let pendingTool = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        // value is Uint8Array → decode to string, add to buffer
        buffer += decoder.decode(value, { stream: true });

        // SSE messages end with \n\n — split on that to get complete events
        const parts = buffer.split("\n\n");
        // Keep the last (potentially incomplete) part in the buffer
        buffer = parts.pop();

        for (const part of parts) {
          // Each SSE line looks like: "data: {...json...}"
          const line = part.trim();
          if (!line.startsWith("data: ")) continue;

          try {
            const data = JSON.parse(line.slice(6)); // strip "data: " prefix

            if (data.type === "done") {
              setIsRunning(false);
              return;
            }

            if (data.type === "error") {
              setError(data.content);
              setIsRunning(false);
              return;
            }

            if (data.type === "tool_call") {
              // Store so we can match with the result that comes next
              pendingTool = { tool: data.tool, args: data.args, result: null };
              setEvents((prev) => [...prev, { ...pendingTool, id: Date.now() }]);
            } else if (data.type === "tool_result") {
              // Update the last tool_call event with its result
              setEvents((prev) => {
                const updated = [...prev];
                for (let i = updated.length - 1; i >= 0; i--) {
                  if (updated[i].tool === data.tool && updated[i].result === null) {
                    updated[i] = { ...updated[i], result: data.content };
                    break;
                  }
                }
                return updated;
              });
            } else {
              // thinking or final_answer
              setEvents((prev) => [...prev, { ...data, id: Date.now() }]);
            }

            // Auto-scroll to the latest event
            setTimeout(() => bottomRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
          } catch {
            // Malformed JSON — skip
          }
        }
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setIsRunning(false);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6 space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900">Research Agent</h1>
        <p className="text-sm text-gray-500 mt-1">
          Searches arXiv papers, the web, and your uploaded documents
        </p>
      </div>

      {/* Question input */}
      <form onSubmit={handleSubmit} className="flex gap-3">
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="What are the latest advances in RAG systems?"
          disabled={isRunning}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-indigo-500 disabled:bg-gray-50"
        />
        <button
          type="submit"
          disabled={isRunning || !question.trim()}
          className="px-5 py-2 bg-indigo-600 text-white rounded-lg font-medium hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isRunning ? "Researching..." : "Ask"}
        </button>
      </form>

      {/* Error display */}
      {error && (
        <div className="px-4 py-3 bg-red-50 border border-red-200 rounded-lg text-red-700 text-sm">
          Error: {error}
        </div>
      )}

      {/* Agent events stream */}
      {events.length > 0 && (
        <div className="space-y-3">
          {/* Thinking spinner */}
          {isRunning && (
            <div className="flex items-center gap-2 text-gray-500 text-sm">
              <span className="animate-spin">⟳</span>
              <span>Agent is working...</span>
            </div>
          )}

          {events.map((event, i) => {
            if (event.type === "thinking") {
              return (
                <div key={event.id ?? i} className="flex gap-3">
                  <span className="text-lg flex-shrink-0">🤔</span>
                  <div className="text-gray-600 italic text-sm py-1">{event.content}</div>
                </div>
              );
            }

            if (event.tool) {
              return (
                <ToolCallBadge
                  key={event.id ?? i}
                  tool={event.tool}
                  args={event.args}
                  result={event.result}
                />
              );
            }

            if (event.type === "final_answer") {
              return <FinalAnswer key={event.id ?? i} content={event.content} />;
            }

            return null;
          })}

          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
