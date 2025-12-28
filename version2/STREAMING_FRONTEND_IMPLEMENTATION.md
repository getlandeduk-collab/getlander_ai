# Frontend Streaming Implementation Guide

## Overview
This guide shows how to implement Server-Sent Events (SSE) streaming in the Chrome extension frontend to get real-time updates during job matching analysis.

## Backend Streaming Endpoint
The backend provides `/api/match-jobs/stream` which streams progress updates as Server-Sent Events (SSE).

## Event Types
The backend sends these event types:
- `status`: Status updates (parsing, scoring, summarizing)
- `job_start`: When a job starts being scored
- `token`: Individual tokens from LLM streaming (real-time text generation)
- `job_complete`: When a job finishes scoring
- `complete`: Final response with all data
- `done`: End of stream
- `error`: Error messages

## Implementation

### 1. Update background.ts

Add this new message handler for streaming:

```typescript
// Add to chrome.runtime.onMessage.addListener in background.ts

if (msg.action === "ANALYZE_JOB_MATCH_STREAM") {
  if (!(await isAuthenticated()).isAuth) {
    return sendResponse({ success: false, error: "Login required" });
  }

  const cookie = await chrome.cookies.get({
    url: "https://get-landed.vercel.app/",
    name: "ext_auth",
  });

  if (!cookie) {
    return sendResponse({ success: false, error: "No auth cookie" });
  }

  try {
    const idToken = cookie.value;
    const payload = JSON.parse(atob(idToken.split(".")[1]));
    const userId = payload.sub ?? payload.user_id;

    /* ---- build pdf Blob ---- */
    const resumeBuffer = new Uint8Array(msg.resumeBuffer);
    const resumeBlob = new Blob([resumeBuffer], {
      type: "application/pdf",
    });

    const fd = new FormData();
    fd.append("pdf_file", resumeBlob, msg.resumeName);

    const jsonBody = {
      jobs: {
        jobtitle: msg.jobData.title,
        joblink: msg.jobData.url,
        jobdata: msg.jobData.jobData,
      },
      user_id: userId,
    };

    fd.append("json_body", JSON.stringify(jsonBody));

    // Use streaming endpoint
    const res = await fetch(
      "https://frightfully-prescholastic-stephine.ngrok-free.dev/api/match-jobs/stream",
      {
        method: "POST",
        body: fd,
      }
    );

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`HTTP ${res.status}: ${errText}`);
    }

    // Set up EventSource-like reader for SSE
    const reader = res.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error("Response body is not readable");
    }

    // Create a port for streaming updates
    const port = chrome.runtime.connect({ name: "streaming" });
    
    let buffer = "";
    let currentJobIndex = 0;
    let currentJobTokens = "";

    while (true) {
      const { done, value } = await reader.read();
      
      if (done) {
        port.postMessage({ type: "done" });
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; // Keep incomplete line in buffer

      for (const line of lines) {
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
            
            // Forward all events to the frontend via port
            port.postMessage(data);
            
            // Handle specific event types
            if (data.type === "job_start") {
              currentJobIndex = data.job_index;
              currentJobTokens = "";
            } else if (data.type === "token") {
              currentJobTokens += data.content;
              // Send accumulated tokens periodically
              if (currentJobTokens.length > 50) {
                port.postMessage({
                  type: "token_batch",
                  job_index: currentJobIndex,
                  content: currentJobTokens,
                });
                currentJobTokens = "";
              }
            } else if (data.type === "complete") {
              // Send final response
              sendResponse({ success: true, data: data.response });
              return;
            } else if (data.type === "error") {
              sendResponse({ success: false, error: data.error });
              return;
            }
          } catch (e) {
            console.error("Failed to parse SSE data:", e, line);
          }
        }
      }
    }

    // Return port for ongoing updates
    return { success: true, port };
  } catch (e: any) {
    console.error("ANALYZE_JOB_MATCH_STREAM failed:", e);
    sendResponse({
      success: false,
      error: e.message || String(e),
    });
  }

  return;
}
```

### 2. Update JobMatcherFeature.tsx

Add streaming state and handlers:

```typescript
// Add to component state
const [streamingStatus, setStreamingStatus] = useState<string>("");
const [streamingProgress, setStreamingProgress] = useState<{
  currentJob?: number;
  totalJobs?: number;
  jobTitle?: string;
  tokens?: string;
}>({});
const [streamingPort, setStreamingPort] = useState<chrome.runtime.Port | null>(null);

// Add streaming handler
const handleAnalyzeStream = () => {
  if (!selectedResume || !currentJob) return;

  let base64 = selectedResume.content;
  if (base64.startsWith("PDF_BASE64:"))
    base64 = base64.replace("PDF_BASE64:", "");

  let buffer: Uint8Array;
  try {
    const binary = atob(base64);
    buffer = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) buffer[i] = binary.charCodeAt(i);
  } catch {
    buffer = new TextEncoder().encode(selectedResume.content);
  }

  const cleanName = selectedResume.name.replace(/\s*\([^)]*\)/g, "").trim();
  
  // Set loading state
  setLoading(true);
  setStreamingStatus("Initializing...");
  setBackendData(null);

  // Set up port listener for streaming updates
  const port = chrome.runtime.connect({ name: "streaming_listener" });
  setStreamingPort(port);

  port.onMessage.addListener((msg: any) => {
    switch (msg.type) {
      case "status":
        setStreamingStatus(msg.message || msg.status);
        break;
      
      case "job_start":
        setStreamingProgress({
          currentJob: msg.job_index,
          totalJobs: msg.total_jobs,
          jobTitle: msg.job_title,
          tokens: "",
        });
        setStreamingStatus(`Analyzing: ${msg.job_title} at ${msg.company}`);
        break;
      
      case "token":
      case "token_batch":
        setStreamingProgress((prev) => ({
          ...prev,
          tokens: (prev.tokens || "") + msg.content,
        }));
        break;
      
      case "job_complete":
        setStreamingStatus(
          `✓ Completed: ${msg.job_title} (${Math.round(msg.match_score * 100)}% match)`
        );
        break;
      
      case "complete":
        setBackendData(msg.response);
        setLoading(false);
        setStreamingStatus("");
        setStreamingProgress({});
        port.disconnect();
        setStreamingPort(null);
        break;
      
      case "error":
        setLoading(false);
        setStreamingStatus(`Error: ${msg.error}`);
        port.disconnect();
        setStreamingPort(null);
        break;
      
      case "done":
        port.disconnect();
        setStreamingPort(null);
        break;
    }
  });

  // Start the analysis
  chrome.runtime.sendMessage(
    {
      action: "ANALYZE_JOB_MATCH_STREAM",
      resumeBuffer: Array.from(buffer),
      resumeName: `${cleanName}.pdf`,
      jobData: currentJob,
    },
    (response) => {
      if (!response?.success && response?.error) {
        setLoading(false);
        setStreamingStatus(`Error: ${response.error}`);
        if (port) {
          port.disconnect();
          setStreamingPort(null);
        }
      }
    }
  );
};

// Cleanup port on unmount
useEffect(() => {
  return () => {
    if (streamingPort) {
      streamingPort.disconnect();
    }
  };
}, [streamingPort]);

// Update the analyze button to use streaming
<Button
  variant="primary"
  size="full"
  className="h-12 text-base shadow-xl shadow-blue-500/20"
  disabled={!selectedResume || !currentJob || loading}
  busy={loading}
  onClick={handleAnalyzeStream} // Use streaming version
  leadingIcon={<Sparkles className="w-5 h-5" />}
>
  {loading ? (
    streamingStatus || "Analyzing Job Match..."
  ) : (
    "Run AI Analysis"
  )}
</Button>

// Add streaming progress indicator
{loading && streamingStatus && (
  <Card className="mt-4 border-blue-200 bg-blue-50/50">
    <CardContent className="p-4">
      <div className="flex items-center gap-3">
        <RefreshCw className="w-4 h-4 text-blue-600 animate-spin" />
        <div className="flex-1">
          <p className="text-sm font-medium text-blue-900">{streamingStatus}</p>
          {streamingProgress.currentJob && (
            <p className="text-xs text-blue-700 mt-1">
              Job {streamingProgress.currentJob} of {streamingProgress.totalJobs}
              {streamingProgress.jobTitle && `: ${streamingProgress.jobTitle}`}
            </p>
          )}
          {streamingProgress.tokens && (
            <div className="mt-2 p-2 bg-white rounded border border-blue-100">
              <p className="text-xs text-gray-600 font-mono">
                {streamingProgress.tokens.slice(-100)}...
              </p>
            </div>
          )}
        </div>
      </div>
    </CardContent>
  </Card>
)}
```

## Alternative: Simpler Fetch-Based Approach

If you prefer a simpler approach without ports, you can use fetch with ReadableStream:

```typescript
// In background.ts - simpler version
if (msg.action === "ANALYZE_JOB_MATCH_STREAM") {
  // ... (same setup code as above)
  
  const res = await fetch(
    "https://frightfully-prescholastic-stephine.ngrok-free.dev/api/match-jobs/stream",
    {
      method: "POST",
      body: fd,
    }
  );

  if (!res.ok) {
    throw new Error(`HTTP ${res.status}`);
  }

  const reader = res.body?.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finalResponse: any = null;

  while (true) {
    const { done, value } = await reader!.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      if (line.startsWith("data: ")) {
        try {
          const data = JSON.parse(line.slice(6));
          
          // Send progress updates via callback
          if (msg.onProgress && typeof msg.onProgress === "function") {
            msg.onProgress(data);
          }
          
          if (data.type === "complete") {
            finalResponse = data.response;
          } else if (data.type === "error") {
            throw new Error(data.error);
          }
        } catch (e) {
          console.error("Parse error:", e);
        }
      }
    }
  }

  sendResponse({ success: true, data: finalResponse });
  return;
}
```

## Benefits of Streaming

1. **Real-time feedback**: Users see progress as it happens
2. **Better UX**: No long waits without feedback
3. **Token-by-token updates**: See LLM generating responses in real-time
4. **Status updates**: Know exactly what stage the analysis is at
5. **Error handling**: Immediate error feedback

## Testing

1. Test with a single job first
2. Verify all event types are received
3. Check that final response matches non-streaming version
4. Test error scenarios (network errors, invalid data)

