import { create } from "zustand";
import { ChatMessage, ModelName } from "../api/types";

interface ChatState {
  messages: ChatMessage[];
  currentAgent: string | null;
  currentModel: ModelName | null;
  threadId: string | null;
  addMessage: (message: ChatMessage) => void;
  clearMessages: () => void;
  setCurrentAgent: (agent: string) => void;
  setCurrentModel: (model: ModelName | null) => void;
  setThreadId: (threadId: string | null) => void;
}

export const useChatStore = create<ChatState>((set) => ({
  messages: [],
  currentAgent: null,
  currentModel: null,
  threadId: null,
  addMessage: (message) =>
    set((state) => ({
      messages: [...state.messages, message],
    })),
  clearMessages: () => set({ messages: [] }),
  setCurrentAgent: (agent) => set({ currentAgent: agent }),
  setCurrentModel: (model) => set({ currentModel: model }),
  setThreadId: (threadId) => set({ threadId }),
}));

