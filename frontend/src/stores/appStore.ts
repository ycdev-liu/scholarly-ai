import { create } from "zustand";

interface AppState {
  apiBaseUrl: string;
  authSecret: string | null;
  theme: "light" | "dark";
  setApiBaseUrl: (url: string) => void;
  setAuthSecret: (secret: string | null) => void;
  setTheme: (theme: "light" | "dark") => void;
}

export const useAppStore = create<AppState>((set) => ({
  apiBaseUrl: import.meta.env.VITE_API_BASE_URL || "http://localhost:8080",
  authSecret: import.meta.env.VITE_AUTH_SECRET || null,
  theme: "light",
  setApiBaseUrl: (url) => set({ apiBaseUrl: url }),
  setAuthSecret: (secret) => set({ authSecret: secret }),
  setTheme: (theme) => set({ theme }),
}));

