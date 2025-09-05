import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Zap, Brain, Target, BarChart3 } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { useToast } from '@/hooks/use-toast';
import AnimatedBackground from '@/components/AnimatedBackground';
import FileUpload from '@/components/FileUpload';
import AnalysisResults from '@/components/AnalysisResults';

interface AnalysisData {
  match_score: number;
  missing_skills: string[];
  highlighted_skills: string[];
  tailored_resume: string;
  cover_letter: string;
  suggestions: string[];
}

const Index = () => {
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescFile, setJobDescFile] = useState<File | null>(null);
  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { toast } = useToast();

  const handleAnalysis = async () => {
    if (!resumeFile || !jobDescFile) {
      toast({
        title: "Missing Files",
        description: "Please upload both resume and job description files.",
        variant: "destructive",
      });
      return;
    }

    setIsAnalyzing(true);
    
    try {
      const formData = new FormData();
      formData.append('resume', resumeFile);
      formData.append('job_description', jobDescFile);

      // Replace with your actual backend URL
      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setAnalysisData(data);
      
      toast({
        title: "Analysis Complete!",
        description: "Your resume analysis is ready.",
      });
    } catch (error) {
      toast({
        title: "Analysis Failed",
        description: "Please try again or check your backend connection.",
        variant: "destructive",
      });
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const features = [
    {
      icon: Brain,
      title: "AI-Powered Analysis",
      description: "Advanced ML algorithms analyze your resume against job requirements"
    },
    {
      icon: Target,
      title: "Skills Matching",
      description: "Identify missing skills and highlight your strengths"
    },
    {
      icon: BarChart3,
      title: "Performance Metrics",
      description: "Get detailed scores and improvement suggestions"
    },
    {
      icon: Zap,
      title: "Instant Results",
      description: "Generate tailored resumes and cover letters in seconds"
    }
  ];

  return (
    <div className="min-h-screen relative">
      <AnimatedBackground />
      
      <div className="relative z-10 container mx-auto px-4 py-8">
        {!analysisData ? (
          <>
            {/* Hero Section */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              className="text-center mb-16"
            >
              <motion.h1 
                className="text-6xl font-bold mb-6 text-gradient-primary"
                initial={{ opacity: 0, y: 20, scale: 0.8 }}
                animate={{ 
                  opacity: 1, 
                  y: 0, 
                  scale: 1,
                  textShadow: [
                    "0 0 0px hsl(var(--primary))",
                    "0 0 10px hsl(var(--primary))",
                    "0 0 0px hsl(var(--primary))"
                  ]
                }}
                transition={{ 
                  delay: 0.2,
                  duration: 0.8,
                  type: "spring",
                  stiffness: 100,
                  textShadow: {
                    duration: 2,
                    repeat: Infinity,
                    ease: "easeInOut"
                  }
                }}
                whileHover={{
                  scale: 1.05,
                  y: -5,
                  transition: { duration: 0.3 }
                }}
              >
                <motion.span
                  animate={{
                    backgroundPosition: ["0% 50%", "100% 50%", "0% 50%"]
                  }}
                  transition={{
                    duration: 3,
                    repeat: Infinity,
                    ease: "linear"
                  }}
                  className="bg-gradient-to-r from-primary via-primary-glow to-primary bg-[length:200%_auto] bg-clip-text text-transparent"
                >
                  JobLens
                </motion.span>
              </motion.h1>
              <motion.p 
                className="text-2xl font-medium text-foreground mb-8 max-w-2xl mx-auto"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 }}
              >
                Transform your job application with AI-powered resume analysis. 
                Get instant feedback, skill matching, and personalized recommendations.
              </motion.p>
              
              <motion.div
                initial={{ opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.6 }}
                className="flex justify-center"
              >
                <Button 
                  variant="gradient" 
                  size="lg"
                  className="text-lg px-8 py-4"
                  onClick={() => {
                    document.getElementById('upload-section')?.scrollIntoView({ 
                      behavior: 'smooth' 
                    });
                  }}
                >
                  Start Analysis
                </Button>
              </motion.div>
            </motion.div>

            {/* Features Grid */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.8 }}
              className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-16"
            >
              {features.map((feature, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: 0.8 + index * 0.1 }}
                  whileHover={{ scale: 1.05 }}
                  className="glass-card p-6 text-center hover-lift"
                >
                  <motion.div
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: 1, opacity: 1 }}
                    transition={{ 
                      delay: 1 + index * 0.2,
                      duration: 0.5,
                      type: "spring",
                      stiffness: 260,
                      damping: 20
                    }}
                    whileHover={{ 
                      scale: 1.2,
                      rotate: [0, -10, 10, -10, 0],
                      transition: { duration: 0.6 }
                    }}
                    className="mb-4"
                  >
                    <motion.div
                      animate={{ 
                        y: [0, -8, 0],
                        scale: [1, 1.05, 1]
                      }}
                      transition={{
                        duration: 2,
                        repeat: Infinity,
                        delay: index * 0.5,
                        ease: "easeInOut"
                      }}
                    >
                      <feature.icon className="h-12 w-12 text-primary mx-auto" />
                    </motion.div>
                  </motion.div>
                  <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                  <p className="text-sm text-muted-foreground">{feature.description}</p>
                </motion.div>
              ))}
            </motion.div>

            {/* Upload Section */}
            <motion.div
              id="upload-section"
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 1 }}
              className="max-w-4xl mx-auto"
            >
              <Card className="glass-card border-primary/20">
                <CardHeader className="text-center">
                  <CardTitle className="text-2xl">Upload Your Documents</CardTitle>
                  <CardDescription>
                    Upload your resume and the job description to get started
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <FileUpload
                      label="Resume"
                      onFileSelect={setResumeFile}
                      selectedFile={resumeFile}
                    />
                    <FileUpload
                      label="Job Description"
                      onFileSelect={setJobDescFile}
                      selectedFile={jobDescFile}
                    />
                  </div>
                  
                  <div className="text-center pt-4">
                    <Button
                      variant="gradient"
                      size="lg"
                      onClick={handleAnalysis}
                      disabled={!resumeFile || !jobDescFile || isAnalyzing}
                      className="px-8"
                    >
                      {isAnalyzing ? (
                        <motion.div
                          animate={{ rotate: 360 }}
                          transition={{ duration: 1, repeat: Infinity, ease: "linear" }}
                          className="w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-2"
                        />
                      ) : (
                        <Zap className="w-5 h-5 mr-2" />
                      )}
                      {isAnalyzing ? 'Analyzing...' : 'Analyze Resume'}
                    </Button>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </>
        ) : (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="max-w-7xl mx-auto"
          >
            <div className="flex items-center justify-between mb-8">
              <div>
                <h1 className="text-4xl font-bold text-gradient-primary">Analysis Results</h1>
                <p className="text-muted-foreground">Your personalized resume analysis report</p>
              </div>
              <Button 
                variant="glass" 
                onClick={() => {
                  setAnalysisData(null);
                  setResumeFile(null);
                  setJobDescFile(null);
                }}
              >
                New Analysis
              </Button>
            </div>
            
            <AnalysisResults data={analysisData} />
          </motion.div>
        )}
      </div>
    </div>
  );
};

export default Index;
