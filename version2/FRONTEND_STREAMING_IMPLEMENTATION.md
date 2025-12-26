# Frontend Streaming Implementation Guide

This guide shows how to implement streaming for the `/api/match-jobs/stream` endpoint in your frontend.

## Backend Endpoint

**Endpoint:** `POST /api/match-jobs/stream`

**Request Format:** Same as `/api/match-jobs` endpoint
- `json_body`: Form data with JSON string
- `pdf_file`: Optional file upload

**Response:** Server-Sent Events (SSE) stream with the following event types:

### Event Types

1. **`status`** - Status updates during processing
   ```json
   {
     "type": "status",
     "status": "parsing|parsing_resume|extracting_jobs|scoring|summarizing",
     "message": "Human-readable message",
     "request_id": "unique-request-id",
     "jobs_count": 5  // Optional
   }
   ```

2. **`job_start`** - When scoring starts for a job
   ```json
   {
     "type": "job_start",
     "job_index": 1,
     "total_jobs": 3,
     "job_title": "Front-End React Developer",
     "company": "Robert Half"
   }
   ```

3. **`token`** - Streaming tokens from LLM (real-time scoring progress)
   ```json
   {
     "type": "token",
     "job_index": 1,
     "content": "{\"match_score\": 0.75,"
   }
   ```

4. **`job_complete`** - When scoring completes for a job
   ```json
   {
     "type": "job_complete",
     "job_index": 1,
     "match_score": 0.75,
     "job_title": "Front-End React Developer"
   }
   ```

5. **`complete`** - Final response with all matched jobs
   ```json
   {
     "type": "complete",
     "response": {
       "candidate_profile": {...},
       "matched_jobs": [...],
       "processing_time": "45.2s",
       "jobs_analyzed": 1,
       "request_id": "abc123",
       "sponsorship": {...}
     }
   }
   ```

6. **`done`** - Stream finished
   ```json
   {
     "type": "done"
   }
   ```

7. **`error`** - Error occurred
   ```json
   {
     "type": "error",
     "error": "Error message",
     "traceback": "..."  // Optional
   }
   ```

---

## Frontend Implementation

### Option 1: Using Fetch API with ReadableStream

```javascript
async function streamMatchJobs(requestData) {
  const formData = new FormData();
  formData.append('json_body', JSON.stringify(requestData));
  
  // If you have a PDF file
  if (requestData.pdf_file) {
    formData.append('pdf_file', requestData.pdf_file);
  }
  
  const response = await fetch('/api/match-jobs/stream', {
    method: 'POST',
    body: formData,
  });
  
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }
  
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  
  // State management
  let currentJobIndex = null;
  let streamingContent = '';
  let matchedJobs = [];
  let finalResponse = null;
  
  while (true) {
    const { done, value } = await reader.read();
    
    if (done) break;
    
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || ''; // Keep incomplete line in buffer
    
    for (const line of lines) {
      if (line.startsWith('data: ')) {
        try {
          const data = JSON.parse(line.slice(6)); // Remove 'data: ' prefix
          
          switch (data.type) {
            case 'status':
              console.log(`Status: ${data.status} - ${data.message}`);
              updateUIStatus(data.status, data.message);
              break;
              
            case 'job_start':
              currentJobIndex = data.job_index;
              streamingContent = '';
              console.log(`Starting job ${data.job_index}/${data.total_jobs}: ${data.job_title}`);
              updateUIJobStart(data);
              break;
              
            case 'token':
              if (data.job_index === currentJobIndex) {
                streamingContent += data.content;
                // Update UI with streaming content (optional - for showing real-time progress)
                updateUIStreamingContent(data.job_index, streamingContent);
              }
              break;
              
            case 'job_complete':
              console.log(`Job ${data.job_index} complete: ${data.match_score}`);
              updateUIJobComplete(data);
              break;
              
            case 'complete':
              finalResponse = data.response;
              matchedJobs = data.response.matched_jobs;
              console.log('Final response received:', finalResponse);
              updateUIFinalResponse(finalResponse);
              break;
              
            case 'done':
              console.log('Stream complete');
              updateUIComplete();
              break;
              
            case 'error':
              console.error('Error:', data.error);
              updateUIError(data.error);
              break;
          }
        } catch (e) {
          console.error('Error parsing SSE data:', e, line);
        }
      }
    }
  }
  
  return finalResponse;
}

// UI Update Functions (customize based on your UI framework)
function updateUIStatus(status, message) {
  // Update status indicator
  document.getElementById('status').textContent = message;
  document.getElementById('status').className = `status-${status}`;
}

function updateUIJobStart(data) {
  // Show job being processed
  const jobElement = document.createElement('div');
  jobElement.id = `job-${data.job_index}`;
  jobElement.innerHTML = `
    <h3>${data.job_title} at ${data.company}</h3>
    <div class="progress">Processing...</div>
  `;
  document.getElementById('jobs-container').appendChild(jobElement);
}

function updateUIStreamingContent(jobIndex, content) {
  // Optional: Show streaming content in real-time
  const jobElement = document.getElementById(`job-${jobIndex}`);
  if (jobElement) {
    const contentDiv = jobElement.querySelector('.streaming-content');
    if (contentDiv) {
      contentDiv.textContent = content.substring(0, 200) + '...';
    }
  }
}

function updateUIJobComplete(data) {
  const jobElement = document.getElementById(`job-${data.job_index}`);
  if (jobElement) {
    jobElement.querySelector('.progress').textContent = `Match Score: ${(data.match_score * 100).toFixed(1)}%`;
    jobElement.querySelector('.progress').className = `progress complete score-${Math.floor(data.match_score * 10)}`;
  }
}

function updateUIFinalResponse(response) {
  // Render final matched jobs
  const container = document.getElementById('results-container');
  container.innerHTML = '';
  
  response.matched_jobs.forEach(job => {
    const jobCard = document.createElement('div');
    jobCard.className = 'job-card';
    jobCard.innerHTML = `
      <h2>${job.job_title}</h2>
      <p class="company">${job.company}</p>
      <p class="score">Match: ${(job.match_score * 100).toFixed(1)}%</p>
      <p class="summary">${job.summary}</p>
      <div class="key-matches">
        <strong>Key Matches:</strong>
        <ul>
          ${job.key_matches.map(match => `<li>${match}</li>`).join('')}
        </ul>
      </div>
    `;
    container.appendChild(jobCard);
  });
  
  // Show sponsorship info if available
  if (response.sponsorship) {
    const sponsorshipDiv = document.createElement('div');
    sponsorshipDiv.className = 'sponsorship-info';
    sponsorshipDiv.innerHTML = `
      <h3>Visa Sponsorship</h3>
      <p>${response.sponsorship.summary}</p>
    `;
    container.appendChild(sponsorshipDiv);
  }
}

function updateUIComplete() {
  document.getElementById('status').textContent = 'Complete!';
  document.getElementById('status').className = 'status-complete';
}

function updateUIError(error) {
  document.getElementById('status').textContent = `Error: ${error}`;
  document.getElementById('status').className = 'status-error';
}
```

### Option 2: Using EventSource (Simpler, but requires GET or special handling)

**Note:** EventSource only supports GET requests. For POST with EventSource, you'll need to use a library or implement a workaround.

### Option 3: React Hook Implementation

```jsx
import { useState, useEffect, useCallback } from 'react';

function useStreamingMatchJobs() {
  const [status, setStatus] = useState('idle');
  const [message, setMessage] = useState('');
  const [currentJob, setCurrentJob] = useState(null);
  const [streamingContent, setStreamingContent] = useState('');
  const [matchedJobs, setMatchedJobs] = useState([]);
  const [finalResponse, setFinalResponse] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  
  const streamMatchJobs = useCallback(async (requestData) => {
    setIsLoading(true);
    setError(null);
    setStatus('starting');
    setMatchedJobs([]);
    setFinalResponse(null);
    
    try {
      const formData = new FormData();
      formData.append('json_body', JSON.stringify(requestData));
      
      if (requestData.pdf_file) {
        formData.append('pdf_file', requestData.pdf_file);
      }
      
      const response = await fetch('/api/match-jobs/stream', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';
      let currentJobIndex = null;
      let jobStreamingContent = '';
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;
        
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';
        
        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              switch (data.type) {
                case 'status':
                  setStatus(data.status);
                  setMessage(data.message);
                  break;
                  
                case 'job_start':
                  currentJobIndex = data.job_index;
                  jobStreamingContent = '';
                  setCurrentJob(data);
                  break;
                  
                case 'token':
                  if (data.job_index === currentJobIndex) {
                    jobStreamingContent += data.content;
                    setStreamingContent(jobStreamingContent);
                  }
                  break;
                  
                case 'job_complete':
                  setCurrentJob(prev => ({ ...prev, complete: true, score: data.match_score }));
                  break;
                  
                case 'complete':
                  setFinalResponse(data.response);
                  setMatchedJobs(data.response.matched_jobs);
                  setStatus('complete');
                  break;
                  
                case 'done':
                  setIsLoading(false);
                  break;
                  
                case 'error':
                  setError(data.error);
                  setStatus('error');
                  setIsLoading(false);
                  break;
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (err) {
      setError(err.message);
      setStatus('error');
      setIsLoading(false);
    }
  }, []);
  
  return {
    streamMatchJobs,
    status,
    message,
    currentJob,
    streamingContent,
    matchedJobs,
    finalResponse,
    error,
    isLoading,
  };
}

// Usage in component
function MatchJobsComponent() {
  const {
    streamMatchJobs,
    status,
    message,
    currentJob,
    streamingContent,
    matchedJobs,
    finalResponse,
    error,
    isLoading,
  } = useStreamingMatchJobs();
  
  const handleSubmit = async (formData) => {
    await streamMatchJobs(formData);
  };
  
  return (
    <div>
      <div className={`status status-${status}`}>
        {message || status}
      </div>
      
      {currentJob && (
        <div className="current-job">
          <h3>Processing: {currentJob.job_title}</h3>
          {streamingContent && (
            <div className="streaming-content">
              {streamingContent.substring(0, 200)}...
            </div>
          )}
        </div>
      )}
      
      {matchedJobs.length > 0 && (
        <div className="matched-jobs">
          {matchedJobs.map(job => (
            <div key={job.rank} className="job-card">
              <h2>{job.job_title}</h2>
              <p>Match: {(job.match_score * 100).toFixed(1)}%</p>
              <p>{job.summary}</p>
            </div>
          ))}
        </div>
      )}
      
      {error && (
        <div className="error">
          Error: {error}
        </div>
      )}
    </div>
  );
}
```

### Option 4: Vue.js Composition API

```vue
<template>
  <div>
    <div :class="`status status-${status}`">
      {{ message || status }}
    </div>
    
    <div v-if="currentJob" class="current-job">
      <h3>Processing: {{ currentJob.job_title }}</h3>
      <div v-if="streamingContent" class="streaming-content">
        {{ streamingContent.substring(0, 200) }}...
      </div>
    </div>
    
    <div v-if="matchedJobs.length > 0" class="matched-jobs">
      <div v-for="job in matchedJobs" :key="job.rank" class="job-card">
        <h2>{{ job.job_title }}</h2>
        <p>Match: {{ (job.match_score * 100).toFixed(1) }}%</p>
        <p>{{ job.summary }}</p>
      </div>
    </div>
    
    <div v-if="error" class="error">
      Error: {{ error }}
    </div>
  </div>
</template>

<script setup>
import { ref } from 'vue';

const status = ref('idle');
const message = ref('');
const currentJob = ref(null);
const streamingContent = ref('');
const matchedJobs = ref([]);
const finalResponse = ref(null);
const error = ref(null);
const isLoading = ref(false);

async function streamMatchJobs(requestData) {
  isLoading.value = true;
  error.value = null;
  status.value = 'starting';
  matchedJobs.value = [];
  finalResponse.value = null;
  
  try {
    const formData = new FormData();
    formData.append('json_body', JSON.stringify(requestData));
    
    if (requestData.pdf_file) {
      formData.append('pdf_file', requestData.pdf_file);
    }
    
    const response = await fetch('/api/match-jobs/stream', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let currentJobIndex = null;
    let jobStreamingContent = '';
    
    while (true) {
      const { done, value } = await reader.read();
      
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const data = JSON.parse(line.slice(6));
            
            switch (data.type) {
              case 'status':
                status.value = data.status;
                message.value = data.message;
                break;
                
              case 'job_start':
                currentJobIndex = data.job_index;
                jobStreamingContent = '';
                currentJob.value = data;
                break;
                
              case 'token':
                if (data.job_index === currentJobIndex) {
                  jobStreamingContent += data.content;
                  streamingContent.value = jobStreamingContent;
                }
                break;
                
              case 'job_complete':
                currentJob.value = { ...currentJob.value, complete: true, score: data.match_score };
                break;
                
              case 'complete':
                finalResponse.value = data.response;
                matchedJobs.value = data.response.matched_jobs;
                status.value = 'complete';
                break;
                
              case 'done':
                isLoading.value = false;
                break;
                
              case 'error':
                error.value = data.error;
                status.value = 'error';
                isLoading.value = false;
                break;
            }
          } catch (e) {
            console.error('Error parsing SSE data:', e);
          }
        }
      }
    }
  } catch (err) {
    error.value = err.message;
    status.value = 'error';
    isLoading.value = false;
  }
}
</script>
```

---

## Example Usage

```javascript
// Example request data
const requestData = {
  user_id: "user123",
  jobs: {
    jobtitle: "Front-End React Developer",
    joblink: "https://example.com/job/123",
    jobdata: "Job description text here..."
  }
};

// Call streaming function
streamMatchJobs(requestData)
  .then(response => {
    console.log('Final response:', response);
    // response contains: candidate_profile, matched_jobs, processing_time, etc.
  })
  .catch(error => {
    console.error('Error:', error);
  });
```

---

## CSS Styling Example

```css
.status {
  padding: 10px;
  border-radius: 4px;
  margin-bottom: 20px;
}

.status-parsing {
  background-color: #e3f2fd;
  color: #1976d2;
}

.status-scoring {
  background-color: #fff3e0;
  color: #f57c00;
}

.status-complete {
  background-color: #e8f5e9;
  color: #388e3c;
}

.status-error {
  background-color: #ffebee;
  color: #d32f2f;
}

.current-job {
  padding: 15px;
  background-color: #f5f5f5;
  border-radius: 4px;
  margin-bottom: 20px;
}

.streaming-content {
  font-family: monospace;
  font-size: 12px;
  color: #666;
  margin-top: 10px;
  max-height: 100px;
  overflow-y: auto;
}

.job-card {
  border: 1px solid #ddd;
  border-radius: 4px;
  padding: 20px;
  margin-bottom: 15px;
}

.job-card .score {
  font-size: 24px;
  font-weight: bold;
  color: #1976d2;
}
```

---

## Key Points

1. **Real-time Updates**: The `token` events allow you to show real-time progress as the LLM generates the scoring response
2. **Status Tracking**: Use `status` events to show progress through different stages
3. **Job Progress**: `job_start` and `job_complete` events let you show progress for each job
4. **Final Response**: The `complete` event contains the full response matching the non-streaming endpoint format
5. **Error Handling**: Always handle `error` events and check for stream completion

This implementation provides a much better user experience by showing progress in real-time rather than waiting for the entire response.

