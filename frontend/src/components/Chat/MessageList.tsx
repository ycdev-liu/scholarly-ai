import { Box, Paper, Typography, Chip, CircularProgress } from "@mui/material";
import type { ChatMessage } from "../../api/types";
import { useEffect, useRef } from "react";

interface MessageListProps {
  messages: ChatMessage[];
  isStreaming?: boolean;
}

export function MessageList({ messages, isStreaming = false }: MessageListProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isStreaming]);

  return (
    <Box sx={{ flex: 1, overflowY: "auto", p: 2 }}>
      {messages.length === 0 && (
        <Box
          sx={{
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            height: "100%",
            color: "text.secondary",
          }}
        >
          <Typography variant="body1">开始对话...</Typography>
        </Box>
      )}
      {messages.map((message, index) => (
        <Box
          key={index}
          sx={{
            display: "flex",
            justifyContent: message.type === "human" ? "flex-end" : "flex-start",
            mb: 2,
          }}
        >
          <Paper
            elevation={1}
            sx={{
              p: 2,
              maxWidth: "70%",
              backgroundColor:
                message.type === "human"
                  ? "primary.main"
                  : message.type === "tool"
                  ? "secondary.light"
                  : "background.paper",
              color: message.type === "human" ? "primary.contrastText" : "text.primary",
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", mb: 1 }}>
              <Chip
                label={message.type.toUpperCase()}
                size="small"
                sx={{ mr: 1 }}
              />
              {message.run_id && (
                <Typography variant="caption" sx={{ opacity: 0.7 }}>
                  {message.run_id.slice(0, 8)}
                </Typography>
              )}
            </Box>
            <Typography variant="body1" sx={{ whiteSpace: "pre-wrap" }}>
              {message.content}
            </Typography>
            {message.tool_calls && message.tool_calls.length > 0 && (
              <Box sx={{ mt: 1 }}>
                {message.tool_calls.map((toolCall, idx) => (
                  <Chip
                    key={idx}
                    label={`Tool: ${toolCall.name}`}
                    size="small"
                    sx={{ mr: 0.5, mb: 0.5 }}
                  />
                ))}
              </Box>
            )}
          </Paper>
        </Box>
      ))}
      {isStreaming && (
        <Box sx={{ display: "flex", justifyContent: "flex-start", mb: 2 }}>
          <Paper elevation={1} sx={{ p: 2 }}>
            <CircularProgress size={20} sx={{ mr: 1 }} />
            <Typography variant="body2" component="span">
              正在生成...
            </Typography>
          </Paper>
        </Box>
      )}
      <div ref={messagesEndRef} />
    </Box>
  );
}

