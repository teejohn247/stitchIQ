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
        console.log('[Pattern] Generating pattern sheet via PatternEngine...');
        const sketchResponse = await fetch(`${AI_WORKER_URL}/worker/pattern-sketches`, {
          method: 'POST',
          headers: workerHeaders(),
          body: JSON.stringify({
            draft_cuts:  result.draft_cuts || [],
            silhouette:  result.specs?.silhouette || result.silhouette || "",
            fabric:      result.specs?.fabric     || result.fabric     || "",
            style_name:  result.specs?.style_name || result.style_name || "",
            sleeves:     result.specs?.sleeves    || result.sleeves    || "Sleeveless",
            uk_size:     12
          })
        });

        if (!sketchResponse.ok) {
          const detail = await workerErrorMessage(sketchResponse);
          console.warn(`[Pattern] Failed: status ${sketchResponse.status} — ${detail}`);
        } else {
          const sketchData = await sketchResponse.json();

          // Store full pattern sheet outputs at the top level
          if (sketchData.pattern_svg)     result.pattern_svg     = sketchData.pattern_svg;
          if (sketchData.pattern_pdf_b64) result.pattern_pdf_b64 = sketchData.pattern_pdf_b64;
          if (sketchData.pattern_json)    result.pattern_json    = sketchData.pattern_json;
          if (sketchData.piece_count)     result.piece_count     = sketchData.piece_count;

          // Also merge per-piece thumbnails for the card grid
          if (sketchData.sketches?.length) {
            const sketchMap: Record<string, string> = {};
            for (const s of sketchData.sketches) {
              sketchMap[s.label.toUpperCase().trim()] = s.svg;
            }
            result.draft_cuts = (result.draft_cuts || []).map((cut: any) => ({
              ...cut,
              svg: sketchMap[cut.label.toUpperCase().trim()] || null
            }));
          }
          console.log(`[Pattern] Done — ${sketchData.piece_count ?? '?'} pieces, `
                    + `sheet SVG: ${sketchData.pattern_svg ? 'yes' : 'no'}`);
        }
      } catch (err: any) {
        console.error('[Pattern] Error generating pattern sheet:', err.message);
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
