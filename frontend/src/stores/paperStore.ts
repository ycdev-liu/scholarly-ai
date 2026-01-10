import { create } from "zustand";

interface Paper {
  id: string;
  title: string;
  authors?: string[];
  abstract?: string;
  arxivId?: string;
  url?: string;
  downloaded?: boolean;
}

interface PaperState {
  papers: Paper[];
  searchResults: Paper[];
  downloadedPapers: Paper[];
  addPaper: (paper: Paper) => void;
  setSearchResults: (papers: Paper[]) => void;
  markAsDownloaded: (paperId: string) => void;
}

export const usePaperStore = create<PaperState>((set) => ({
  papers: [],
  searchResults: [],
  downloadedPapers: [],
  addPaper: (paper) =>
    set((state) => ({
      papers: [...state.papers, paper],
    })),
  setSearchResults: (papers) => set({ searchResults: papers }),
  markAsDownloaded: (paperId) =>
    set((state) => ({
      downloadedPapers: [
        ...state.downloadedPapers,
        ...state.searchResults.filter((p) => p.id === paperId),
      ],
      searchResults: state.searchResults.map((p) =>
        p.id === paperId ? { ...p, downloaded: true } : p
      ),
    })),
}));

