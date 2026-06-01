import { Request, Response } from 'express';
import mongoose from 'mongoose';
import { v2 as cloudinary } from 'cloudinary';
import { Job } from '../models/Job';
import { enqueueAiJob } from '../services/queue';
import { workerHeaders, workerErrorMessage } from '../services/workerClient';

cloudinary.config({
  cloud_name: process.env.CLOUDINARY_CLOUD_NAME,
  api_key: process.env.CLOUDINARY_API_KEY,
  api_secret: process.env.CLOUDINARY_API_SECRET
});

const uploadImageToCloudinary = async (base64Image: string): Promise<string> => {
  try {
    const uploadRes = await cloudinary.uploader.upload(base64Image, {
      folder: 'stitchiq/uploads',
    });
    return uploadRes.secure_url;
  } catch (error: any) {
    console.error('Cloudinary upload failed:', error.message);
    throw new Error(`Failed to upload image to Cloudinary: ${error.message}`);
  }
};

const getAiWorkerUrl = () => process.env.AI_WORKER_URL || 'http://127.0.0.1:8000';
const getWorkerToken = () => process.env.WORKER_TOKEN || 'your-secret-token';

// Centralised in-memory database fallback when MongoDB is down
export const inMemoryJobs = new Map<string, any>();

// Helper to check database connectivity
const isDbConnected = () => mongoose.connection.readyState === 1;

/**
 * Direct worker trigger fallback if Redis/Queue is offline or DB is disconnected.
 * Invokes the FastAPI Python Worker directly and updates the job status.
 */
async function runInMemoryJob(jobId: string, type: string, inputData: any) {
  const job = inMemoryJobs.get(jobId) || { id: jobId, type, inputData, status: 'pending', createdAt: new Date() };
  job.status = 'processing';
  inMemoryJobs.set(jobId, job);

  if (isDbConnected()) {
    try {
      await Job.findByIdAndUpdate(jobId, { status: 'processing' });
    } catch (e) {
      console.warn('Failed to update status to processing in MongoDB:', e);
    }
  }

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
        image_url: inputData.imageUrl || 'https://res.cloudinary.com/demo/image/upload/v1312461204/sample.jpg',
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

    console.log(`[Resilience Engine] Hitting AI worker directly at ${getAiWorkerUrl()}${endpoint}`);
    const response = await fetch(`${getAiWorkerUrl()}${endpoint}`, {
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
        console.log('[Sketches] Fetching AI sketches for pattern pieces in direct fallback...');
        const sketchResponse = await fetch(`${getAiWorkerUrl()}/worker/pattern-sketches`, {
          method: 'POST',
          headers: workerHeaders(),
          body: JSON.stringify({
            draft_cuts: result.draft_cuts || [],
            silhouette: result.specs?.silhouette || "",
            fabric: result.specs?.fabric || ""
          })
        });
        if (sketchResponse.ok) {
          const sketchData = await sketchResponse.json();
          const sketchMap: Record<string, string> = {};
          for (const s of sketchData.sketches || []) {
            sketchMap[s.label.toUpperCase().trim()] = s.svg;
          }
          result.draft_cuts = (result.draft_cuts || []).map((cut: any) => ({
            ...cut,
            svg: sketchMap[cut.label.toUpperCase().trim()] || null
          }));
          console.log('[Sketches] Successfully merged AI sketches in direct fallback.');
        } else {
          console.warn(`[Sketches] Failed to fetch sketches in direct fallback: status ${sketchResponse.status}`);
        }
      } catch (err: any) {
        console.error('[Sketches] Error fetching sketches in direct fallback:', err.message);
      }
    }
    
    // Update local memory
    job.status = 'completed';
    job.resultData = result;
    inMemoryJobs.set(jobId, job);
    
    // Update DB if connected
    if (isDbConnected()) {
      await Job.findByIdAndUpdate(jobId, { 
        status: 'completed',
        resultData: result
      });
    }
    console.log(`[Resilience Engine] Job ${jobId} finished processing successfully.`);
  } catch (err: any) {
    console.error(`[Resilience Engine] Background job ${jobId} failed:`, err.message);
    job.status = 'failed';
    job.resultData = { error: err.message };
    inMemoryJobs.set(jobId, job);

    if (isDbConnected()) {
      try {
        await Job.findByIdAndUpdate(jobId, { 
          status: 'failed',
          resultData: { error: err.message }
        });
      } catch (e) {}
    }
  }
}

export const createJob = async (req: Request, res: Response): Promise<void> => {
  try {
    const { type, inputData } = req.body;
    
    if (!['pattern_analysis', 'virtual_tryon', 'occasion_stylist'].includes(type)) {
      res.status(400).json({ error: 'Invalid job type' });
      return;
    }

    // Intercept base64 images and upload to Cloudinary so we send clean URLs to Colab
    if (inputData && inputData.imageUrl && inputData.imageUrl.startsWith('data:image')) {
      console.log('[Cloudinary] Base64 image detected in inputData.imageUrl. Uploading...');
      inputData.imageUrl = await uploadImageToCloudinary(inputData.imageUrl);
      console.log(`[Cloudinary] Successfully uploaded: ${inputData.imageUrl}`);
    }

    if (inputData && inputData.maskUrl && inputData.maskUrl.startsWith('data:image')) {
      console.log('[Cloudinary] Base64 mask detected in inputData.maskUrl. Uploading...');
      inputData.maskUrl = await uploadImageToCloudinary(inputData.maskUrl);
      console.log(`[Cloudinary] Successfully uploaded: ${inputData.maskUrl}`);
    }

    let jobId: string;
    
    if (isDbConnected()) {
      const job = new Job({
        type,
        inputData,
        status: 'pending'
      });
      await job.save();
      jobId = job._id.toString();
    } else {
      // Offline fallback id
      jobId = 'mem_' + Math.random().toString(36).substring(2, 15);
      const mockJob = {
        id: jobId,
        type,
        inputData,
        status: 'pending',
        createdAt: new Date()
      };
      inMemoryJobs.set(jobId, mockJob);
      console.log(`[Resilience Engine] MongoDB offline. Saved job ${jobId} to memory.`);
    }

    // Try to queue
    let enqueued = false;
    if (isDbConnected()) {
      enqueued = await enqueueAiJob(jobId, type, inputData);
    }
    
    // Direct worker trigger fallback if Redis is down or DB is offline
    if (!isDbConnected() || !enqueued) {
      console.log(`[Resilience Engine] Redis/MongoDB offline. Triggering direct asynchronous execution.`);
      runInMemoryJob(jobId, type, inputData);
    }

    res.status(201).json({ 
      message: 'Job created successfully',
      jobId: jobId 
    });
  } catch (error) {
    console.error('Error creating job:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};

export const getJobStatus = async (req: Request, res: Response): Promise<void> => {
  try {
    const { id } = req.params;
    const jobId = id as string;
    
    // Check in memory first
    if (inMemoryJobs.has(jobId)) {
      const job = inMemoryJobs.get(jobId);
      res.status(200).json(job);
      return;
    }

    if (!isDbConnected()) {
      res.status(404).json({ error: 'Database is offline and job not found in memory.' });
      return;
    }
    
    const job = await Job.findById(id);
    
    if (!job) {
      res.status(404).json({ error: 'Job not found' });
      return;
    }

    res.status(200).json({
      id: job._id,
      status: job.status,
      type: job.type,
      resultData: job.resultData,
      createdAt: job.createdAt
    });
  } catch (error) {
    console.error('Error fetching job:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
};
