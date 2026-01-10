import { Box, Paper, Typography, Alert, Button, Stack } from "@mui/material";
import { useState, useEffect } from "react";
import { ChatWindow } from "../Chat/ChatWindow";
import { useChatStore } from "../../stores/chatStore";
import { apiClient } from "../../api/client";

export function PaperSearch() {
  const { setCurrentAgent } = useChatStore();
  const [error, setError] = useState<string | null>(null);

  // 设置默认使用 paper-research-supervisor agent
  useEffect(() => {
    apiClient
      .getInfo()
      .then((info) => {
        const paperAgent = info.agents.find((a) => a.key.includes("paper"));
        if (paperAgent) {
          setCurrentAgent(paperAgent.key);
        }
      })
      .catch((err) => {
        setError(`无法连接到服务: ${err.message}`);
      });
  }, [setCurrentAgent]);

  return (
    <Box sx={{ height: "calc(100vh - 120px)", display: "flex", flexDirection: "column" }}>
      <Paper elevation={1} sx={{ p: 2, mb: 2 }}>
        <Typography variant="h6" gutterBottom>
          论文搜索与下载
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          使用 AI Agent 搜索 OpenReview 和 arXiv 上的学术论文，并下载到本地。
        </Typography>
        <Stack direction="row" spacing={2} sx={{ mt: 2 }}>
          <Button
            variant="outlined"
            onClick={() => {
              // 触发搜索示例
            }}
          >
            搜索示例：Transformer 论文
          </Button>
          <Button
            variant="outlined"
            onClick={() => {
              // 触发下载示例
            }}
          >
            下载示例：arXiv:1706.03762
          </Button>
        </Stack>
      </Paper>
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}
      <Box sx={{ flex: 1 }}>
        <ChatWindow />
      </Box>
    </Box>
  );
}

