/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        heading: ['"Bricolage Grotesque"', 'sans-serif'],
        serif: ['"Instrument Serif"', 'Georgia', 'serif'],
        mono: ['"JetBrains Mono"', 'ui-monospace', 'SFMono-Regular', 'Menlo', 'monospace'],
        sans: ['"Bricolage Grotesque"', 'sans-serif'],
      },
      colors: {
        blue: {
          brand: '#0F45E3',
        },
        lime: {
          brand: '#C8FF00',
        },
        navy: {
          brand: '#0B1A5C',
        },
        paper: {
          brand: '#FFF9F7',
        },
        verdict: {
          fake: '#FF4D4D',
          real: '#1FD17A',
        },
      },
      boxShadow: {
        'hard-navy-sm': '3px 3px 0px #0B1A5C',
        'hard-navy': '6px 6px 0px #0B1A5C',
        'hard-navy-lg': '8px 8px 0px #0B1A5C',
        'hard-navy-xl': '12px 12px 0px #0B1A5C',
        'hard-lime': '4px 4px 0px #C8FF00',
        'hard-white': '5px 5px 0px #FFFFFF',
      },
      letterSpacing: {
        'tightest': '-0.04em',
        'tighter': '-0.025em',
        'widest-mono': '0.12em',
      }
    },
  },
  plugins: [],
}
