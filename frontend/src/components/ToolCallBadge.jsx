// Maps tool names to human-readable labels and colors
const TOOL_CONFIG = {
  arxiv_search: { label: "arXiv Search", color: "bg-purple-100 text-purple-800 border-purple-200" },
  web_search: { label: "Web Search", color: "bg-blue-100 text-blue-800 border-blue-200" },
  document_store_search: { label: "Document Store", color: "bg-green-100 text-green-800 border-green-200" },
};

export default function ToolCallBadge({ tool, args, result }) {
  const config = TOOL_CONFIG[tool] || {
    label: tool,
    color: "bg-gray-100 text-gray-800 border-gray-200",
  };

  // Show the first arg value as the query preview
  const queryPreview = args ? Object.values(args)[0] : null;

  return (
    <div className="ml-6 my-2 border rounded-lg overflow-hidden text-sm">
      <div className={`flex items-center gap-2 px-3 py-2 border-b font-medium ${config.color}`}>
        <span>🔧</span>
        <span>{config.label}</span>
        {queryPreview && (
          <span className="font-normal opacity-75 truncate max-w-xs">
            "{queryPreview}"
          </span>
        )}
      </div>
      {result && (
        <div className="px-3 py-2 bg-gray-50 text-gray-600 text-xs max-h-32 overflow-y-auto whitespace-pre-wrap">
          {result}
        </div>
      )}
    </div>
  );
}
