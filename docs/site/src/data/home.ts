export type IconName =
  | 'chat'
  | 'document'
  | 'globe'
  | 'voice'
  | 'local'
  | 'ownership'
  | 'browser'
  | 'open'
  | 'server'
  | 'access'
  | 'integration';

export interface CardData {
  icon: IconName;
  title: string;
  body: string;
}

export interface StepData {
  title: string;
  body: string;
  command?: string;
}

export interface EnterpriseTileData {
  icon: IconName;
  title: string;
  body: string;
}

export interface FaqData {
  question: string;
  answer: string;
}

export const features: CardData[] = [
  {
    icon: 'chat',
    title: 'Chat and create',
    body: 'Ask questions, draft content, plan work, and get streaming responses in one browser workspace.'
  },
  {
    icon: 'document',
    title: 'Ask your documents',
    body: 'Upload PDFs, notes, reports, and spreadsheets. Hybrid search combines semantic and keyword retrieval before answering.'
  },
  {
    icon: 'globe',
    title: 'Search the web',
    body: 'Add live web results when an answer needs current information, with sources kept beside the response.'
  },
  {
    icon: 'voice',
    title: 'Work by voice',
    body: 'Use microphone input or the realtime WebRTC voice pipeline for hands-free conversations.'
  }
];

export const benefits: CardData[] = [
  {
    icon: 'local',
    title: 'Local first, cloud optional',
    body: 'Run Ollama on your machine. OpenRouter and Nvidia NIM remain available when you choose a hosted model.'
  },
  {
    icon: 'ownership',
    title: 'Clear data boundaries',
    body: 'Self-hosted authentication and per-user isolation keep conversations, sessions, and RAG contexts separated.'
  },
  {
    icon: 'browser',
    title: 'A familiar interface',
    body: 'Once the stack is running, use LiteMindUI from any modern browser at http://localhost:3000.'
  },
  {
    icon: 'open',
    title: 'Open for inspection',
    body: 'Read the source, run it yourself, and adapt the FastAPI backend or Next.js frontend to your needs.'
  }
];

export const steps: StepData[] = [
  {
    title: 'Run the installer',
    body: 'Pull the prebuilt images and start the Docker stack.',
    command: 'curl -fsSL https://raw.githubusercontent.com/debabratamishra/litemind-ui/main/install.sh | bash'
  },
  {
    title: 'Open your workspace',
    body: 'Visit http://localhost:3000, create an account, and sign in.'
  },
  {
    title: 'Choose a model',
    body: 'Connect Ollama for local inference or configure a hosted provider.'
  }
];

export const enterpriseTiles: EnterpriseTileData[] = [
  {
    icon: 'server',
    title: 'Private deployment',
    body: 'Run the application inside your network and keep model, document, and identity services under your control.'
  },
  {
    icon: 'access',
    title: 'Identity and access',
    body: 'Adapt the self-hosted authentication layer and per-user data isolation to your access model.'
  },
  {
    icon: 'integration',
    title: 'Custom integrations',
    body: 'Connect internal tools, private models, document stores, and business-specific workflows.'
  }
];

export const faqs: FaqData[] = [
  {
    question: 'Which operating systems are supported?',
    answer: 'LiteMindUI runs through Docker on systems that support the provided Compose setup. The browser interface works in current Chrome, Edge, Firefox, and Safari releases.'
  },
  {
    question: 'Can it run without the internet?',
    answer: 'Yes, after dependencies and a local model are available. Offline use requires Ollama or another local inference service; web search and hosted models need network access.'
  },
  {
    question: 'Where is my data stored?',
    answer: 'Your deployment controls storage. The canonical conversation and user-memory stores use PostgreSQL, while document and ingestion paths are configured on the host.'
  },
  {
    question: 'Do I need an OpenRouter or Nvidia account?',
    answer: 'No. Ollama provides the local model path. Hosted providers are optional and require their own API credentials.'
  },
  {
    question: 'Is LiteMindUI ready for a company rollout?',
    answer: 'The open-source project provides the base application, self-hosted authentication, and per-user isolation. Production deployments should be reviewed against your security, support, and compliance requirements.'
  }
];
