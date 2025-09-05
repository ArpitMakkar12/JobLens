import axios from 'axios';

// Get the base URL from the environment variables set on Render/Vercel
const BASE_URL = import.meta.env.VITE_API_URL;

// Define the shape of the API response
export interface AnalysisResult {
  match_score: number;
  missing_skills: string[];
  highlighted_skills: string[];
  tailored_resume: string;
  cover_letter: string;
  suggestions: string[]; // ✅ Correctly defined as an array
}

export const analyzeDocuments = async (
  resumeFile: File,
  jdFile: File
): Promise<AnalysisResult> => {
  // This check provides a clear error if the environment variable is missing
  if (!BASE_URL) {
    throw new Error(
      'VITE_API_URL is not defined. Check your deployment environment variables.'
    );
  }

  const API_URL = `${BASE_URL}/analyze`;
  const formData = new FormData();
  formData.append('resume', resumeFile);
  formData.append('job_description', jdFile);

  const response = await axios.post<AnalysisResult>(API_URL, formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};