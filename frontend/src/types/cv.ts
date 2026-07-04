export interface EducationItem {
  institution: string | null;
  degree: string | null;
  field_of_study: string | null;
  start_date: string | null;
  end_date: string | null;
}

export interface ExperienceItem {
  company: string | null;
  title: string | null;
  start_date: string | null;
  end_date: string | null;
  description: string | null;
}

export interface CVExtraction {
  name: string | null;
  email: string | null;
  phone: string | null;
  skills: string[];
  education: EducationItem[];
  experience: ExperienceItem[];
  current_role: string | null;
  domain: string | null;
  extra_fields: Record<string, string>;
}

export interface CVRecord {
  id: string;
  filename: string;
  extraction: CVExtraction;
  raw_text: string;
}

export interface RetrievedCV {
  id: string;
  filename: string;
  score: number;
  extraction: CVExtraction;
}

export interface QueryResponse {
  answer: string;
  best_candidate_id: string | null;
  best_candidate_reason: string | null;
  retrieved: RetrievedCV[];
}
