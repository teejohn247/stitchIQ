import Bull from 'bull';
import { Job as JobModel } from '../models/Job';
import { workerHeaders, workerErrorMessage } from './workerClient';

const REDIS_URL = process.env.REDIS_URL || 'redis://127.0.0.1:6379';
const AI_WORKER_URL = process.env.AI_WORKER_URL || 'http://127.0.0.1:8000';
const WORKER_TOKEN = process.env.WORKER_TOKEN || 'your-secret-token';

export const aiJobQueue = new Bull('ai-jobs', REDIS_URL, {
  redis: {
    maxRetriesPerRequest: 1, // Fail fast if Redis is not running
  }
});

aiJobQueue.on('error', (error) => {
  console.warn('Bull queue error (Redis might not be running):', error.message);
});

// Process jobs (routes tasks to the Python worker)
aiJobQueue.process(async (job) => {
  const { jobId, type, inputData } = job.data;

  // Mark as processing
  await JobModel.findByIdAndUpdate(jobId, { status: 'processing' });

  try {
    let endpoint = '/worker/sdxl';
    let payload = {};

    if (type === 'pattern_analysis') {
      endpoint = '/worker/pattern-analysis';
      payload = {
        image_url: inputData.imageUrl
      };
    } else if (type === 'virtual_tryon') {
      endpoint = '/worker/sdxl-inpaint';
      payload = {
        image_url: inputData.imageUrl,
        mask_url: inputData.maskUrl || "",
        prompt: inputData.prompt || 'try on garment'
      };
    } else if (type === 'occasion_stylist') {
      endpoint = '/worker/sdxl';
      payload = {
        prompt: inputData.prompt || 'African fashion occasion stylist outfit'
      };
    } else if (type === 'fabric_pricing') {
      endpoint = '/worker/fabric-price';
      payload = {
        prompt: inputData.prompt || 'African fashion fabric',
        fabric_name: inputData.fabricName,
        base_price: inputData.basePrice || 10000
      };
    }

    const response = await fetch(`${AI_WORKER_URL}${endpoint}`, {
      method: 'POST',
      headers: workerHeaders(),
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const detail = await workerErrorMessage(response);
      throw new Error(`AI worker returned status ${response.status}: ${detail}`);
    }

    let result = await response.json();

    if (type === 'pattern_analysis') {
      try {
        console.log('[Sketches] Fetching AI sketches for pattern pieces in queue...');
        const sketchResponse = await fetch(`${AI_WORKER_URL}/worker/pattern-sketches`, {
          method: 'POST',
          headers: workerHeaders(),
          body: JSON.stringify({
            draft_cuts: result.draft_cuts || [],
            silhouette: result.specs?.silhouette || "",
            fabric: result.specs?.fabric || ""
          })
        });
        if (!sketchResponse.ok) {
          const detail = await workerErrorMessage(sketchResponse);
          console.warn(`[Sketches] Failed: status ${sketchResponse.status} — ${detail}`);
        }
        if (sketchResponse.ok) {
          const sketchData = await sketchResponse.json();
          // Build map with normalised keys (uppercase, trimmed) so label
          // differences between Gemini and Claude don't break the merge.
          const sketchMap: Record<string, string> = {};
          for (const s of sketchData.sketches || []) {
            sketchMap[s.label.toUpperCase().trim()] = s.svg;
          }
          result.draft_cuts = (result.draft_cuts || []).map((cut: any) => ({
            ...cut,
            svg: sketchMap[cut.label.toUpperCase().trim()] || null
          }));
          console.log('[Sketches] Successfully merged AI sketches in queue.');
        } else {
          console.warn(`[Sketches] Failed to fetch sketches in queue: status ${sketchResponse.status}`);
        }
      } catch (err: any) {
        console.error('[Sketches] Error fetching sketches in queue:', err.message);
      }
    }

    // Mark as completed with Python worker's result
    await JobModel.findByIdAndUpdate(jobId, {
      status: 'completed',
      resultData: result
    });

  } catch (err: any) {
    console.error(`AI Job ${jobId} failed:`, err.message);
    await JobModel.findByIdAndUpdate(jobId, {
      status: 'failed',
      resultData: { error: err.message }
    });
  }

  return { success: true };
});

export const enqueueAiJob = async (jobId: string, type: string, inputData: any) => {
  try {
    await aiJobQueue.add({ jobId, type, inputData });
    return true;
  } catch (err: any) {
    console.error('Failed to enqueue job:', err.message);
    return false;
  }
};
