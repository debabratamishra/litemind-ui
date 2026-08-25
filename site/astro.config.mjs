// @ts-check
import { defineConfig } from 'astro/config';
import react from '@astrojs/react';

export default defineConfig({
  site: 'https://debabratamishra.github.io',
  base: '/litemind-ui',
  output: 'static',
  integrations: [react()],
});
