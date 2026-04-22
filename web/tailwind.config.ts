import type { Config } from "tailwindcss";

export default {
  content: ["./src/**/*.{js,ts,jsx,tsx,mdx}"],
  theme: {
    extend: {
      colors: {
        background: "hsl(224 71% 4%)",
        foreground: "hsl(213 31% 91%)",
        card: "hsl(224 71% 6%)",
        border: "hsl(215 28% 17%)",
        muted: "hsl(215 20% 65%)",
        primary: "hsl(234 89% 74%)",
        success: "hsl(142 71% 45%)",
        warning: "hsl(38 92% 50%)",
        danger: "hsl(0 84% 60%)",
      },
    },
  },
  plugins: [],
} satisfies Config;
