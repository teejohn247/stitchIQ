import dotenv from 'dotenv';
dotenv.config();

import express from 'express';
import cors from 'cors';
import mongoose from 'mongoose';
import jobRoutes from './routes/jobRoutes';

const app = express();
const port = process.env.PORT || 3000;
const mongoUri = process.env.MONGO_URI || 'mongodb://localhost:27018/stitchiq';

app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ limit: '50mb', extended: true }));

// Routes
app.use('/api/jobs', jobRoutes);

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'StitchIQ Backend API v2.0',
    dbConnected: mongoose.connection.readyState === 1
  });
});

// Start server regardless of DB connection so /health check works
app.listen(port, () => {
  console.log(`Server is running on port ${port}`);
});

// Try to connect to Database
mongoose.connect(mongoUri)
  .then(() => {
    console.log('Connected to MongoDB');
  })
  .catch((error) => {
    console.warn('MongoDB connection error. Make sure MongoDB is running locally or MONGO_URI is set.');
    console.warn(error.message);
  });
