import type { Config } from "tailwindcss"

const config: Config = {
  darkMode: ["class"],
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        // IDE chrome
        ide: {
          bg:       "hsl(var(--ide-bg))",
          surface:  "hsl(var(--ide-surface))",
          elevated: "hsl(var(--ide-elevated))",
          border:   "hsl(var(--ide-border))",
          sidebar:  "hsl(var(--ide-sidebar))",
        },
        // Semantic
        foreground:  "hsl(var(--foreground))",
        muted:       "hsl(var(--muted))",
        accent:      "hsl(var(--accent))",
        "accent-2":  "hsl(var(--accent-2))",
        danger:      "hsl(var(--danger))",
        success:     "hsl(var(--success))",
        warning:     "hsl(var(--warning))",
        // Tab active indicator
        tab:         "hsl(var(--tab))",
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "ui-monospace", "SFMono-Regular", "monospace"],
        sans: ["'Inter'", "system-ui", "sans-serif"],
      },
      fontSize: {
        "2xs": ["0.65rem", { lineHeight: "1rem" }],
      },
      boxShadow: {
        "panel":  "0 0 0 1px hsl(var(--ide-border))",
        "glow":   "0 0 20px hsl(var(--accent) / 0.25)",
        "glow-danger": "0 0 16px hsl(var(--danger) / 0.3)",
      },
      keyframes: {
        "pulse-dot": {
          "0%, 100%": { opacity: "1" },
          "50%": { opacity: "0.3" },
        },
        "slide-in-left": {
          from: { transform: "translateX(-8px)", opacity: "0" },
          to:   { transform: "translateX(0)",    opacity: "1" },
        },
        "fade-up": {
          from: { transform: "translateY(6px)", opacity: "0" },
          to:   { transform: "translateY(0)",   opacity: "1" },
        },
        "progress-bar": {
          from: { width: "0%" },
          to:   { width: "100%" },
        },
        "shimmer": {
          "0%":   { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
      },
      animation: {
        "pulse-dot":      "pulse-dot 1.4s ease-in-out infinite",
        "slide-in-left":  "slide-in-left 0.2s ease-out",
        "fade-up":        "fade-up 0.3s ease-out",
        "shimmer":        "shimmer 2s linear infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
}

export default config
