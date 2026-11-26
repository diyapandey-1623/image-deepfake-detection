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
        card: '#FFFFFF',
        border: '#DCE3F2',
        text: {
          DEFAULT: '#0B1A5C',
          muted: '#5A678C',
        },
        surface: {
          subtle: '#F4F7FC',
        },
        primary: {
          DEFAULT: '#0F45E3',
          dark: '#0B1A5C',
          light: '#DCE7FF',
        },
        verdict: {
          fake: '#FF4D4D',
          fakeBg: '#FEF2F2',
          fakeBorder: '#FECACA',
          real: '#16A34A',
          realBg: '#F0FDF4',
          realBorder: '#BBF7D0',
          uncertain: '#D97706',
          uncertainBg: '#FFFBEB',
          uncertainBorder: '#FDE68A',
        },
      },
      borderWidth: {
        '3': '3px',
      },
      spacing: {
        '4.5': '1.125rem',
      },
      boxShadow: {
        'subtle': '0 1px 2px rgba(11, 26, 92, 0.06)',
        'modal': '0 10px 25px rgba(11, 26, 92, 0.14)',
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
