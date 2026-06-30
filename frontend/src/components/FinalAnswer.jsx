export default function FinalAnswer({ content }) {
  // Split on double newlines to render paragraphs
  const paragraphs = content.split(/\n\n+/).filter(Boolean);

  return (
    <div className="mt-4 border border-green-200 rounded-lg overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2 bg-green-50 border-b border-green-200 font-medium text-green-800">
        <span>✅</span>
        <span>Final Answer</span>
      </div>
      <div className="px-4 py-3 space-y-3 text-gray-800 leading-relaxed">
        {paragraphs.map((para, i) => (
          <p key={i}>{para}</p>
        ))}
      </div>
    </div>
  );
}
