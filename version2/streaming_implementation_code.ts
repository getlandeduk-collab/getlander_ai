// ============================================================================
// STREAMING IMPLEMENTATION FOR CHROME EXTENSION
// ============================================================================
// Add these code snippets to your existing files

// ============================================================================
// 1. ADD TO background.ts - New streaming message handler
// ============================================================================

// Add this inside chrome.runtime.onMessage.addListener, after ANALYZE_JOB_MATCH handler:

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

    // Set up SSE reader
    const reader = res.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      throw new Error("Response body is not readable");
    }

    // Create port for streaming updates
    const port = chrome.runtime.connect({ name: `stream_${Date.now()}` });
    
    let buffer = "";
    let finalResponse: any = null;

    // Read stream
    (async () => {
      try {
        while (true) {
          const { done, value } = await reader.read();
          
          if (done) {
            port.postMessage({ type: "done" });
            if (finalResponse) {
              sendResponse({ success: true, data: finalResponse });
            }
            break;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            if (line.startsWith("data: ")) {
              try {
                const data = JSON.parse(line.slice(6));
                
                // Forward all events to frontend
                port.postMessage(data);
                
                if (data.type === "complete") {
                  finalResponse = data.response;
                } else if (data.type === "error") {
                  port.postMessage({ type: "error", error: data.error });
                  sendResponse({ success: false, error: data.error });
                  return;
                }
              } catch (e) {
                console.error("Failed to parse SSE data:", e);
              }
            }
          }
        }
      } catch (e: any) {
        port.postMessage({ type: "error", error: e.message });
        sendResponse({ success: false, error: e.message });
      }
    })();

    // Return port name so frontend can connect
    return { success: true, portName: port.name };
  } catch (e: any) {
    console.error("ANALYZE_JOB_MATCH_STREAM failed:", e);
    sendResponse({
      success: false,
      error: e.message || String(e),
    });
  }

  return true; // Keep channel open for async response
}

// ============================================================================
// 2. ADD TO JobMatcherFeature.tsx - Streaming state and handler
// ============================================================================

// Add to component state (near other useState declarations):
const [streamingStatus, setStreamingStatus] = useState<string>("");
const [streamingProgress, setStreamingProgress] = useState<{
  currentJob?: number;
  totalJobs?: number;
  jobTitle?: string;
  company?: string;
  matchScore?: number;
}>({});
const [streamingPort, setStreamingPort] = useState<chrome.runtime.Port | null>(null);

// Add streaming handler function (replace handleAnalyze):
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
  
  // Reset state
  setLoading(true);
  setStreamingStatus("Initializing analysis...");
  setBackendData(null);
  setStreamingProgress({});

  // Send message to background
  chrome.runtime.sendMessage(
    {
      action: "ANALYZE_JOB_MATCH_STREAM",
      resumeBuffer: Array.from(buffer),
      resumeName: `${cleanName}.pdf`,
      jobData: currentJob,
    },
    (response) => {
      if (response?.success && response?.portName) {
        // Connect to streaming port
        const port = chrome.runtime.connect({ name: response.portName });
        setStreamingPort(port);

        port.onMessage.addListener((msg: any) => {
          switch (msg.type) {
            case "status":
              setStreamingStatus(msg.message || msg.status || "Processing...");
              break;
            
            case "job_start":
              setStreamingProgress({
                currentJob: msg.job_index,
                totalJobs: msg.total_jobs,
                jobTitle: msg.job_title,
                company: msg.company,
              });
              setStreamingStatus(`Analyzing: ${msg.job_title} at ${msg.company}`);
              break;
            
            case "token":
              // Optional: show token-by-token generation (can be noisy)
              // You can accumulate tokens here if needed
              break;
            
            case "job_complete":
              setStreamingProgress((prev) => ({
                ...prev,
                matchScore: msg.match_score,
              }));
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
              setStreamingProgress({});
              port.disconnect();
              setStreamingPort(null);
              break;
            
            case "done":
              port.disconnect();
              setStreamingPort(null);
              break;
          }
        });

        port.onDisconnect.addListener(() => {
          setStreamingPort(null);
        });
      } else if (response?.success === false) {
        setLoading(false);
        setStreamingStatus(`Error: ${response.error || "Unknown error"}`);
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

// ============================================================================
// 3. UPDATE THE ANALYZE BUTTON IN JobMatcherFeature.tsx
// ============================================================================

// Replace the existing "Run AI Analysis" button with:

{!backendData && (
  <section className="sticky bottom-0 bg-gradient-to-t from-slate-50 via-slate-50 to-transparent pb-4 pt-2 -mx-4 px-4 z-10 transition-all duration-300">
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
    
    {/* Streaming Progress Indicator */}
    {loading && streamingStatus && (
      <div className="mt-3 p-3 bg-blue-50 border border-blue-200 rounded-lg">
        <div className="flex items-center gap-2">
          <RefreshCw className="w-4 h-4 text-blue-600 animate-spin" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-blue-900 truncate">
              {streamingStatus}
            </p>
            {streamingProgress.currentJob && (
              <p className="text-xs text-blue-700 mt-1">
                Job {streamingProgress.currentJob} of {streamingProgress.totalJobs}
                {streamingProgress.jobTitle && ` • ${streamingProgress.jobTitle}`}
                {streamingProgress.matchScore !== undefined && (
                  <span className="ml-2 font-semibold">
                    ({Math.round(streamingProgress.matchScore * 100)}% match)
                  </span>
                )}
              </p>
            )}
          </div>
        </div>
      </div>
    )}
  </section>
)}

// ============================================================================
// 4. OPTIONAL: Add progress bar for visual feedback
// ============================================================================

// Add this import at the top:
// import { Progress } from "./ui/Progress"; // If you have a Progress component

// Add progress bar in the streaming indicator:
{loading && streamingProgress.currentJob && (
  <div className="mt-2">
    <div className="flex justify-between text-xs text-blue-700 mb-1">
      <span>Progress</span>
      <span>
        {streamingProgress.currentJob} / {streamingProgress.totalJobs}
      </span>
    </div>
    <div className="w-full bg-blue-100 rounded-full h-2">
      <div
        className="bg-blue-600 h-2 rounded-full transition-all duration-300"
        style={{
          width: `${(streamingProgress.currentJob / (streamingProgress.totalJobs || 1)) * 100}%`,
        }}
      />
    </div>
  </div>
)}

// ============================================================================
// USAGE NOTES
// ============================================================================
// 1. Replace handleAnalyze with handleAnalyzeStream
// 2. The streaming version provides real-time updates
// 3. Users see progress as jobs are analyzed
// 4. Final response format is identical to non-streaming version
// 5. All existing UI components work the same way


