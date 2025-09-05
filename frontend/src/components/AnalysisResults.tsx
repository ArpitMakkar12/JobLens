import React from 'react';
import { motion } from 'framer-motion';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  RadialBarChart,
  RadialBar
} from 'recharts';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { CheckCircle, XCircle, Target, TrendingUp, FileText, Mail } from 'lucide-react';

interface AnalysisData {
  match_score: number;
  missing_skills: string[];
  highlighted_skills: string[];
  tailored_resume: string;
  cover_letter: string;
  suggestions: string[];
}

interface AnalysisResultsProps {
  data: AnalysisData;
}

const AnalysisResults: React.FC<AnalysisResultsProps> = ({ data }) => {
  const { match_score, missing_skills, highlighted_skills, tailored_resume, cover_letter, suggestions } = data;

  const pieData = [
    { name: 'Match', value: match_score, color: 'hsl(263 70% 50.4%)' },
    { name: 'Gap', value: 100 - match_score, color: 'hsl(240 3.7% 15.9%)' }
  ];

  const skillsData = [
    { name: 'Highlighted Skills', value: highlighted_skills.length, fill: 'hsl(263 70% 50.4%)' },
    { name: 'Missing Skills', value: missing_skills.length, fill: 'hsl(0 62.8% 30.6%)' }
  ];

  const getScoreColor = (score: number) => {
    if (score >= 75) return 'text-green-400';
    if (score >= 50) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getScoreLabel = (score: number) => {
    if (score >= 75) return 'Excellent Match';
    if (score >= 50) return 'Good Match';
    return 'Needs Improvement';
  };

  return (
    <div className="space-y-6">
      {/* Header with Match Score */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div className="inline-flex items-center gap-4 glass-card p-6 rounded-2xl">
          <div className="text-center">
            <div className={`text-6xl font-bold ${getScoreColor(match_score)}`}>
              {match_score}%
            </div>
            <p className="text-muted-foreground text-sm mt-1">
              {getScoreLabel(match_score)}
            </p>
          </div>
          <div className="w-32 h-32">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={60}
                  dataKey="value"
                  startAngle={90}
                  endAngle={450}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Skills Analysis */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.1 }}
        >
          <Card className="glass-card border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="h-5 w-5 text-primary" />
                Skills Analysis
              </CardTitle>
              <CardDescription>
                Comparison of your skills with job requirements
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={skillsData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(240 3.7% 15.9%)" />
                    <XAxis 
                      dataKey="name" 
                      stroke="hsl(240 5% 64.9%)"
                      fontSize={12}
                    />
                    <YAxis stroke="hsl(240 5% 64.9%)" fontSize={12} />
                    <Tooltip 
                      contentStyle={{
                        backgroundColor: 'hsl(240 10% 3.9%)',
                        border: '1px solid hsl(240 3.7% 15.9%)',
                        borderRadius: '8px'
                      }}
                    />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Skills Breakdown */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
          className="space-y-4"
        >
          {/* Highlighted Skills */}
          <Card className="glass-card border-green-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-green-400">
                <CheckCircle className="h-5 w-5" />
                Matching Skills ({highlighted_skills.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {highlighted_skills.slice(0, 8).map((skill, index) => (
                  <Badge 
                    key={index}
                    variant="secondary" 
                    className="bg-green-500/10 text-green-400 border-green-500/20"
                  >
                    {skill}
                  </Badge>
                ))}
                {highlighted_skills.length > 8 && (
                  <Badge variant="outline" className="text-muted-foreground">
                    +{highlighted_skills.length - 8} more
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Missing Skills */}
          <Card className="glass-card border-red-500/20">
            <CardHeader className="pb-3">
              <CardTitle className="flex items-center gap-2 text-red-400">
                <XCircle className="h-5 w-5" />
                Missing Skills ({missing_skills.length})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2">
                {missing_skills.slice(0, 8).map((skill, index) => (
                  <Badge 
                    key={index}
                    variant="secondary" 
                    className="bg-red-500/10 text-red-400 border-red-500/20"
                  >
                    {skill}
                  </Badge>
                ))}
                {missing_skills.length > 8 && (
                  <Badge variant="outline" className="text-muted-foreground">
                    +{missing_skills.length - 8} more
                  </Badge>
                )}
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>

      {/* Suggestions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card className="glass-card border-primary/20">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="h-5 w-5 text-primary" />
              Improvement Suggestions
            </CardTitle>
            <CardDescription>
              AI-powered recommendations to enhance your application
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {suggestions.map((suggestion, index) => (
                <div key={index} className="flex items-start gap-3 p-3 rounded-lg bg-primary/5 border border-primary/10">
                  <div className="w-2 h-2 rounded-full bg-primary mt-2 flex-shrink-0" />
                  <p className="text-sm text-foreground">{suggestion}</p>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      {/* AI Generated Content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Tailored Resume */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          <Card className="glass-card border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <FileText className="h-5 w-5 text-primary" />
                Tailored Resume Suggestions
              </CardTitle>
              <CardDescription>
                AI-optimized version aligned with the job description
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-muted/30 rounded-lg p-4 max-h-96 overflow-y-auto">
                <pre className="whitespace-pre-wrap text-sm text-foreground font-mono">
                  {tailored_resume}
                </pre>
              </div>
            </CardContent>
          </Card>
        </motion.div>

        {/* Cover Letter */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
        >
          <Card className="glass-card border-primary/20">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Mail className="h-5 w-5 text-primary" />
                Generated Cover Letter
              </CardTitle>
              <CardDescription>
                Personalized cover letter for this position
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="bg-muted/30 rounded-lg p-4 max-h-96 overflow-y-auto">
                <pre className="whitespace-pre-wrap text-sm text-foreground font-mono">
                  {cover_letter}
                </pre>
              </div>
            </CardContent>
          </Card>
        </motion.div>
      </div>
    </div>
  );
};

export default AnalysisResults;