import { create } from "zustand";

interface VectorDB {
  name: string;
  path: string;
  type: string;
  created_at?: string;
  metadata?: Record<string, unknown>;
}

interface VectorDBState {
  databases: VectorDB[];
  currentDatabase: VectorDB | null;
  setDatabases: (databases: VectorDB[]) => void;
  setCurrentDatabase: (db: VectorDB | null) => void;
  addDatabase: (db: VectorDB) => void;
}

export const useVectorDBStore = create<VectorDBState>((set) => ({
  databases: [],
  currentDatabase: null,
  setDatabases: (databases) => set({ databases }),
  setCurrentDatabase: (db) => set({ currentDatabase: db }),
  addDatabase: (db) =>
    set((state) => ({
      databases: [...state.databases, db],
    })),
}));

