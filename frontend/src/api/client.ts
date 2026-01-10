import axios, { type AxiosInstance } from "axios";
import type {
  ChatMessage,
  ChatHistory,
  ChatHistoryInput,
  Feedback,
  FeedbackResponse,
  ServiceMetadata,
  StreamInput,
  UserInput,
  HealthStatus,
  StreamEvent,
} from "./types";

export class ApiClient {
  private client: AxiosInstance;
  private baseURL: string;
  private authSecret?: string;

  constructor(baseURL?: string, authSecret?: string) {
    this.baseURL = baseURL || import.meta.env.VITE_API_BASE_URL || "http://localhost:8080";
    this.authSecret = authSecret || import.meta.env.VITE_AUTH_SECRET;
    this.client = axios.create({
      baseURL: this.baseURL,
      headers: {
        "Content-Type": "application/json",
        ...(this.authSecret && { Authorization: `Bearer ${this.authSecret}` }),
      },
    });
  }

  /**
   * 获取服务元数据
   */
  async getInfo(): Promise<ServiceMetadata> {
    const response = await this.client.get<ServiceMetadata>("/info");
    return response.data;
  }

  /**
   * 健康检查
   */
  async getHealth(): Promise<HealthStatus> {
    const response = await this.client.get<HealthStatus>("/health");
    return response.data;
  }

  /**
   * 同步调用 agent
   */
  async invoke(
    input: UserInput,
    agentId?: string
  ): Promise<ChatMessage> {
    const url = agentId ? `/${agentId}/invoke` : "/invoke";
    const response = await this.client.post<ChatMessage>(url, input);
    return response.data;
  }

  /**
   * 流式调用 agent (SSE)
   */
  async *stream(
    input: StreamInput,
    agentId?: string,
    onEvent?: (event: StreamEvent) => void
  ): AsyncGenerator<StreamEvent, void, unknown> {
    const url = agentId ? `/${agentId}/stream` : "/stream";
    
    const response = await fetch(`${this.baseURL}${url}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(this.authSecret && { Authorization: `Bearer ${this.authSecret}` }),
      },
      body: JSON.stringify(input),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Response body is not readable");
    }

    const decoder = new TextDecoder();
    let buffer = "";

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = line.slice(6);
            if (data === "[DONE]") {
              return;
            }

            try {
              const parsed = JSON.parse(data);
              const event: StreamEvent = {
                type: parsed.type,
                content: parsed.type === "message" ? parsed.content : parsed.content,
              };
              
              if (onEvent) {
                onEvent(event);
              }
              yield event;
            } catch (e) {
              console.error("Error parsing SSE data:", e);
            }
          }
        }
      }
    } finally {
      reader.releaseLock();
    }
  }

  /**
   * 获取聊天历史
   */
  async getHistory(input: ChatHistoryInput): Promise<ChatHistory> {
    const response = await this.client.post<ChatHistory>("/history", input);
    return response.data;
  }

  /**
   * 提交反馈
   */
  async submitFeedback(feedback: Feedback): Promise<FeedbackResponse> {
    const response = await this.client.post<FeedbackResponse>("/feedback", feedback);
    return response.data;
  }
}

// 创建默认实例
export const apiClient = new ApiClient();

