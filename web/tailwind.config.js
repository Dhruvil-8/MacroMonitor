/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './src/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        bg: {
          0: 'var(--bg-0)',
          1: 'var(--bg-1)',
          2: 'var(--bg-2)',
        },
        line: {
          DEFAULT: 'var(--line)',
          strong: 'var(--line-strong)',
        },
        tx: {
          0: 'var(--tx-0)',
          1: 'var(--tx-1)',
          2: 'var(--tx-2)',
        },
        pos: 'var(--pos)',
        neg: 'var(--neg)',
        neutral: 'var(--neutral)',
        warn: 'var(--warn)',
        accent: 'var(--accent)',
      },
      fontFamily: {
        mono: ['"JetBrains Mono"', 'monospace'],
        sans: ['Inter', 'sans-serif'],
      },
    },
  },
  plugins: [],
};
