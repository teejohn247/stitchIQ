const WORKER_TOKEN = process.env.WORKER_TOKEN || 'your-secret-token';

export function workerHeaders(): Record<string, string> {
  return {
    'Content-Type': 'application/json',
    'x-worker-token': WORKER_TOKEN,
    // Required for ngrok free tier programmatic calls
    'ngrok-skip-browser-warning': 'true',
  };
}

export async function workerErrorMessage(response: Response): Promise<string> {
  const body = await response.text();
  try {
    const json = JSON.parse(body);
    if (json.detail) {
      return typeof json.detail === 'string' ? json.detail : JSON.stringify(json.detail);
    }
    if (json.error) return String(json.error);
  } catch {
    // not JSON — fall through
  }
  return body.slice(0, 300) || response.statusText;
}
