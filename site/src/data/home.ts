// Data model for the home page content
export interface CardData {
  icon: string;
  title: string;
  body: string;
}

export interface StepData {
  title: string;
  body: string;
}

export interface EnterpriseTileData {
  icon: string;
  title: string;
  body: string;
}

// Feature cards (first section after hero)
export const features: CardData[] = [
  {
    icon: '💬',
    title: 'Chat',
    body: 'Have a conversation. Ask questions, brainstorm ideas, or get help writing an email, a story, or a plan.'
  },
  {
    icon: '📄',
    title: 'Ask your documents',
    body: 'Drop in your PDFs, notes, or reports and ask questions about them. Ask "What did we decide in the meeting notes?" and it will tell you.'
  },
  {
    icon: '🌐',
    title: 'Search the web',
    body: 'Let it look things up online and bring back a clear, sourced answer instead of a wall of links.'
  },
  {
    icon: '🎙️',
    title: 'Talk out loud',
    body: 'Turn on voice mode and just speak. It listens, thinks, and replies with a natural voice, like a phone call with your assistant.'
  }
];

// Benefit cards (second section)
export const benefits: CardData[] = [
  {
    icon: '🔒',
    title: 'Private by design',
    body: 'Your files and conversations can stay entirely on your own computer. Nothing leaves your machine unless you choose to use a cloud service.'
  },
  {
    icon: '🖥️',
    title: 'Works offline',
    body: 'With a local AI model, LiteMindUI keeps working even without the internet. No connection? No problem.'
  },
  {
    icon: '🤝',
    title: 'Friendly for everyone',
    body: 'You do not need to be a programmer. If you can open a web page, you can use it.'
  },
  {
    icon: '🌱',
    title: 'Open source',
    body: 'The code is free and open for anyone to read, improve, and trust. No lock-in, no surprises.'
  }
];

// Steps for "How do I get started?"
export const steps: StepData[] = [
  {
    title: 'Get a copy',
    body: 'Download or clone the project from GitHub to your computer. It is free.'
  },
  {
    title: 'Run it',
    body: 'One command starts everything (Docker does the heavy lifting). No manual setup needed.'
  },
  {
    title: 'Open & chat',
    body: 'Open the address it shows you in your browser, and start talking to your AI. That\'s it.'
  }
];

// Enterprise section tiles
export const enterpriseTiles: EnterpriseTileData[] = [
  {
    icon: '🛡️',
    title: 'Private, on-prem deployment',
    body: 'Run entirely inside your own network, with no data leaving your perimeter and no third-party APIs required.'
  },
  {
    icon: '🔐',
    title: 'SSO & access control',
    body: 'Connect your identity provider and decide exactly who can see what, down to workspace and document level.'
  },
  {
    icon: '📈',
    title: 'SLAs & priority support',
    body: 'Guaranteed response times and a direct line to me when something business-critical is on the line.'
  },
  {
    icon: '🧩',
    title: 'Custom integrations',
    body: 'Wire up your internal tools, private models, and data sources, plus features built to your spec.'
  }
];