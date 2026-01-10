// API 类型定义，对应后端 schema

export type MessageType = "human" | "ai" | "tool" | "custom";

export type ModelName = "gpt-4o-mini" | "gpt-4o" | "openai-compatible" | "fake";

export interface ToolCall {
  name: string;
  args: Record<string, unknown>;
  id: string | null;
  type?: "tool_call";
}

export interface ChatMessage {
  type: MessageType;
  content: string;
  tool_calls?: ToolCall[];
  tool_call_id?: string | null;
  run_id?: string | null;
  response_metadata?: Record<string, unknown>;
  custom_data?: Record<string, unknown>;
}

export interface AgentInfo {
  key: string;
  description: string;
}

export interface ServiceMetadata {
  agents: AgentInfo[];
  models: ModelName[];
  default_agent: string;
  default_model: ModelName;
}

export interface UserInput {
  message: string;
  model?: ModelName | null;
  thread_id?: string | null;
  user_id?: string | null;
  agent_config?: Record<string, unknown>;
}

export interface StreamInput extends UserInput {
  stream_tokens?: boolean;
}

export interface Feedback {
  run_id: string;
  key: string;
  score: number;
  kwargs?: Record<string, unknown>;
}

export interface FeedbackResponse {
  status: "success";
}

export interface ChatHistoryInput {
  thread_id: string;
}

export interface ChatHistory {
  messages: ChatMessage[];
}

export interface StreamEvent {
  type: "message" | "token" | "error";
  content: ChatMessage | string;
}

export interface HealthStatus {
  status: string;
  langfuse?: string;
  checkpointer?: {
    type: string;
    purpose: string;
    file_exists?: string;
    file_size?: string;
    file_path?: string;
    connection?: string;
  };
  store?: {
    type: string;
    purpose: string;
    connection?: string;
  };
}

