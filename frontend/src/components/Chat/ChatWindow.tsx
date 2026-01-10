import { Box, Paper, Stack, Alert } from "@mui/material";
import { useState, useEffect, useCallback } from "react";
import { MessageList } from "./MessageList";
import { MessageInput } from "./MessageInput";
import { AgentSelector } from "./AgentSelector";
import { ModelSelector } from "./ModelSelector";
import { apiClient } from "../../api/client";
import type { ChatMessage, ServiceMetadata, StreamEvent } from "../../api/types";
import { useChatStore } from "../../stores/chatStore";

export function ChatWindow() {
  const {
    messages,
    currentAgent,
    currentModel,
    threadId,
    addMessage,
    setCurrentAgent,
    setCurrentModel,
    setThreadId,
  } = useChatStore();

  const [serviceInfo, setServiceInfo] = useState<ServiceMetadata | null>(null);
  const [isStreaming, setIsStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentStreamingMessage, setCurrentStreamingMessage] = useState<ChatMessage | null>(null);

  useEffect(() => {
    // 加载服务信息
    apiClient
      .getInfo()
      .then((info) => {
        setServiceInfo(info);
        if (!currentAgent && info.default_agent) {
          setCurrentAgent(info.default_agent);
        }
      })
      .catch((err) => {
        setError(`无法连接到服务: ${err.message}`);
      });
  }, [currentAgent, setCurrentAgent]);

  const handleSend = useCallback(
    async (messageText: string) => {
      if (!currentAgent || !serviceInfo) {
        setError("请先选择 Agent");
        return;
      }

      setError(null);
      setIsStreaming(true);
      setCurrentStreamingMessage(null);

      // 添加用户消息
      const userMessage: ChatMessage = {
        type: "human",
        content: messageText,
      };
      addMessage(userMessage);

      try {
        let accumulatedContent = "";
        let streamedMessage: ChatMessage | null = null;

        for await (const event of apiClient.stream(
          {
            message: messageText,
            model: currentModel || undefined,
            thread_id: threadId || undefined,
            stream_tokens: true,
          },
          currentAgent,
          (event: StreamEvent) => {
            if (event.type === "token" && typeof event.content === "string") {
              accumulatedContent += event.content;
              streamedMessage = {
                type: "ai",
                content: accumulatedContent,
              };
              setCurrentStreamingMessage(streamedMessage);
            } else if (event.type === "message" && typeof event.content !== "string") {
              const msg = event.content as ChatMessage;
              if (msg.type === "ai") {
                streamedMessage = msg;
                setCurrentStreamingMessage(msg);
                if (msg.run_id && !threadId) {
                  // 使用 run_id 作为 thread_id（如果后端支持）
                  setThreadId(msg.run_id);
                }
              }
            } else if (event.type === "error") {
              setError(typeof event.content === "string" ? event.content : "发生错误");
            }
          }
        )) {
          // Stream events are handled in the callback
        }

        // 添加最终消息
        if (streamedMessage) {
          addMessage(streamedMessage);
          setCurrentStreamingMessage(null);
        }
      } catch (err) {
        setError(`发送消息失败: ${err instanceof Error ? err.message : "未知错误"}`);
        addMessage({
          type: "ai",
          content: `错误: ${err instanceof Error ? err.message : "未知错误"}`,
        });
      } finally {
        setIsStreaming(false);
        setCurrentStreamingMessage(null);
      }
    },
    [currentAgent, currentModel, threadId, serviceInfo, addMessage, setThreadId]
  );

  const displayMessages = currentStreamingMessage
    ? [...messages, currentStreamingMessage]
    : messages;

  return (
    <Box sx={{ height: "calc(100vh - 120px)", display: "flex", flexDirection: "column" }}>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      <Stack direction="row" spacing={2} sx={{ mb: 2 }}>
        {serviceInfo && (
          <>
            <AgentSelector
              agents={serviceInfo.agents}
              selectedAgent={currentAgent || serviceInfo.default_agent}
              onAgentChange={setCurrentAgent}
            />
            <ModelSelector
              models={serviceInfo.models}
              selectedModel={currentModel}
              onModelChange={setCurrentModel}
            />
          </>
        )}
      </Stack>
      <Paper
        elevation={1}
        sx={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          mb: 2,
        }}
      >
        <MessageList messages={displayMessages} isStreaming={isStreaming} />
      </Paper>
      <MessageInput onSend={handleSend} disabled={isStreaming || !currentAgent} />
    </Box>
  );
}

