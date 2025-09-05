import React, { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { Upload, FileText, X } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { cn } from '@/lib/utils';

interface FileUploadProps {
  onFileSelect: (file: File) => void;
  selectedFile?: File;
  label: string;
  accept?: string;
  className?: string;
}

const FileUpload: React.FC<FileUploadProps> = ({
  onFileSelect,
  selectedFile,
  label,
  accept = '.pdf,.docx',
  className
}) => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      onFileSelect(acceptedFiles[0]);
    }
  }, [onFileSelect]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'application/pdf': ['.pdf'],
      'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
    },
    multiple: false
  });

  const removeFile = () => {
    onFileSelect(null as any);
  };

  return (
    <div className={cn("w-full", className)}>
      <label className="block text-sm font-medium text-foreground mb-2">
        {label}
      </label>
      
      {selectedFile ? (
        <motion.div
          initial={{ scale: 0.95, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          className="glass-card p-4 border-2 border-primary/50"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <FileText className="h-8 w-8 text-primary" />
              <div>
                <p className="font-medium text-foreground">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground">
                  {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                </p>
              </div>
            </div>
            <Button
              variant="ghost"
              size="icon"
              onClick={removeFile}
              className="text-destructive hover:bg-destructive/10"
            >
              <X className="h-4 w-4" />
            </Button>
          </div>
        </motion.div>
      ) : (
        <div
          {...getRootProps()}
          className={cn(
            "glass-card border-2 border-dashed border-border p-8 text-center cursor-pointer transition-all duration-300",
            isDragActive && "border-primary bg-primary/5 scale-105",
            "hover:border-primary/50 hover:bg-primary/5 hover:scale-105"
          )}
        >
          <input {...getInputProps()} />
          <Upload className="mx-auto h-12 w-12 text-muted-foreground mb-4" />
          <p className="text-lg font-medium text-foreground mb-2">
            {isDragActive ? 'Drop your file here' : 'Upload your file'}
          </p>
          <p className="text-sm text-muted-foreground">
            Drag and drop or click to select • PDF, DOCX supported
          </p>
        </div>
      )}
    </div>
  );
};

export default FileUpload;