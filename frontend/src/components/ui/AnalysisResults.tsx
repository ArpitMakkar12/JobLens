import { AnalysisResult } from '@/lib/api'; // Import the type
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";

interface AnalysisResultsProps {
  results: AnalysisResult;
}

export function AnalysisResults({ results }: AnalysisResultsProps) {
  const getScoreColor = (score: number) => {
    if (score > 75) return 'bg-green-500';
    if (score > 50) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="space-y-8 mt-8">
      <Card>
        <CardHeader>
          <CardTitle className="text-center">Match Score</CardTitle>
        </CardHeader>
        <CardContent className="text-center">
          <p className={`text-6xl font-bold ${getScoreColor(results.match_score).replace('bg-', 'text-')}`}>
            {results.match_score}%
          </p>
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <CardHeader><CardTitle>✅ Matched Skills</CardTitle></CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {results.highlighted_skills.map((skill, i) => <Badge key={i} variant="secondary">{skill}</Badge>)}
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle>❌ Missing Skills</CardTitle></CardHeader>
          <CardContent className="flex flex-wrap gap-2">
            {results.missing_skills.map((skill, i) => <Badge key={i} variant="destructive">{skill}</Badge>)}
          </CardContent>
        </Card>
      </div>

      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="suggestions">
          <AccordionTrigger>💡 AI Suggestions</AccordionTrigger>
          <AccordionContent className="whitespace-pre-wrap">{results.suggestions}</AccordionContent>
        </AccordionItem>
        <AccordionItem value="resume">
          <AccordionTrigger>📄 AI-Tailored Resume</AccordionTrigger>
          <AccordionContent className="whitespace-pre-wrap">{results.tailored_resume}</AccordionContent>
        </AccordionItem>
        <AccordionItem value="cover-letter">
          <AccordionTrigger>✉️ AI-Generated Cover Letter</AccordionTrigger>
          <AccordionContent className="whitespace-pre-wrap">{results.cover_letter}</AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
}