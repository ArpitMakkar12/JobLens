import axios from 'axios';

const API_URL = 'https://skill-sync-backend.onrender.com/analyze'; // ⬅️ Your URL from Render

// Define the shape of the data your API will return
export interface AnalysisResult {
  match_score: number;
  missing_skills: string[];
  highlighted_skills: string[];
  tailored_resume: string;
  cover_letter: string;
  suggestions: string;
}

/**
 * Analyzes the resume and job description by sending them to the backend.
 * @param resumeFile - The user's resume file.
 * @param jdFile - The job description file.
 * @returns A promise that resolves to the analysis results.
 */
export const analyzeDocuments = async (resumeFile: File, jdFile: File): Promise<AnalysisResult> => {
  const formData = new FormData();
  formData.append('resume', resumeFile);
  formData.append('job_description', jdFile);

  try {
    const response = await axios.post<AnalysisResult>(API_URL, formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch (error) {
    console.error("Error during API call:", error);
    // Re-throw the error so the component can handle it
    throw new Error('Failed to get analysis from the server.');
  }
};