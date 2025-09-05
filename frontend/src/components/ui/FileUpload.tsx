import React, { useState } from 'react';
import { Button } from "@/components/ui/button"; // Assuming shadcn/ui
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

// Define the props this component will accept
interface FileUploadProps {
  onAnalyze: (resumeFile: File, jdFile: File) => void;
  isLoading: boolean;
}

export function FileUpload({ onAnalyze, isLoading }: FileUploadProps) {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jdFile, setJdFile] = useState<File | null>(null);

  const handleSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (resumeFile && jdFile) {
      onAnalyze(resumeFile, jdFile);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="grid w-full max-w-sm items-center gap-1.5">
          <Label htmlFor="resume">📄 Your Resume</Label>
          <Input 
            id="resume" 
            type="file" 
            accept=".pdf,.docx"
            required
            onChange={(e) => setResumeFile(e.target.files?.[0] || null)}
          />
        </div>
        <div className="grid w-full max-w-sm items-center gap-1.5">
          <Label htmlFor="job_description">📋 Job Description</Label>
          <Input 
            id="job_description" 
            type="file" 
            accept=".pdf,.docx"
            required
            onChange={(e) => setJdFile(e.target.files?.[0] || null)}
          />
        </div>
      </div>
      <Button type="submit" disabled={isLoading} className="w-full">
        {isLoading ? 'Analyzing...' : 'Analyze Now'}
      </Button>
    </form>
  );
}