import mongoose, { Document, Schema } from 'mongoose';

export interface IJob extends Document {
  type: 'pattern_analysis' | 'virtual_tryon' | 'occasion_stylist';
  status: 'pending' | 'processing' | 'completed' | 'failed';
  inputData: any;
  resultData?: any;
  userId?: mongoose.Types.ObjectId;
  createdAt: Date;
  updatedAt: Date;
}

const JobSchema: Schema = new Schema({
  type: { type: String, enum: ['pattern_analysis', 'virtual_tryon', 'occasion_stylist'], required: true },
  status: { type: String, enum: ['pending', 'processing', 'completed', 'failed'], default: 'pending' },
  inputData: { type: Schema.Types.Mixed, required: true },
  resultData: { type: Schema.Types.Mixed },
  userId: { type: Schema.Types.ObjectId, ref: 'User' },
}, { timestamps: true });

export const Job = mongoose.model<IJob>('Job', JobSchema);
